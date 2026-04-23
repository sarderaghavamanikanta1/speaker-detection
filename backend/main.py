import os, uuid, logging, json, wave
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="SoundSense API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Logging & Environment
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("soundsense")

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ─────────────────────────────────────────────
# Optional ML Libraries
# ─────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import librosa
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    HAS_ML_LIBS = True
except ImportError:
    HAS_ML_LIBS = False
    log.warning("Heavy ML libraries not found. Local model disabled.")

# ─────────────────────────────────────────────
# Optional Gemini AI
# ─────────────────────────────────────────────
try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        log.info("Gemini AI integration enabled ✓")
except ImportError:
    log.warning("google-generativeai not installed.")

UPLOAD_DIR = Path("/tmp") if os.environ.get("VERCEL") else Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

MODEL_PATH   = Path(__file__).parent / "model" / "model.pth"
SAMPLE_RATE  = 16_000
N_MELS       = 64
N_FFT        = 512
HOP_LENGTH   = 512
EMBEDDING_DIM = 64
SEGMENT_SEC  = 0.8
SEGMENT_HOP_SEC = 0.2
MAX_DURATION_SEC = 300


# ─────────────────────────────────────────────
# CNN Architecture  (matches notebook exactly)
#   features.0  Conv2d(1,  8, 3×3) + ReLU + MaxPool2d(2)
#   features.3  Conv2d(8, 16, 3×3) + ReLU + AdaptiveAvgPool2d(4×4)
#   fc.0        Flatten
#   fc.1        Linear(256 → 64)
#   Output      L2-normalized embedding  (NOT a classifier)
# ─────────────────────────────────────────────
if HAS_ML_LIBS:
    class FastSpeakerEmbeddingNet(nn.Module):                  # ← fixed: correct class name & architecture
        def __init__(self, embedding_dim: int = EMBEDDING_DIM):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),     # features.0
                nn.ReLU(),                                      # features.1
                nn.MaxPool2d(2),                                # features.2
                nn.Conv2d(8, 16, kernel_size=3, padding=1),    # features.3
                nn.ReLU(),                                      # features.4
                nn.AdaptiveAvgPool2d((4, 4)),                   # features.5  ← fixed: was MaxPool2d
            )
            self.fc = nn.Sequential(
                nn.Flatten(),                                   # fc.0
                nn.Linear(16 * 4 * 4, embedding_dim),          # fc.1  ← fixed: 256→64, no extra layers
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.unsqueeze(1)                                 # (B, 1, mel, time)
            x = self.features(x)
            x = self.fc(x)
            x = F.normalize(x, p=2, dim=1)                    # ← fixed: L2 norm, not softmax
            return x
else:
    class FastSpeakerEmbeddingNet:
        def __init__(self, *args, **kwargs): pass
        def eval(self): pass
        def __call__(self, *args, **kwargs): return None


# ─────────────────────────────────────────────
# Model Initialization
# ─────────────────────────────────────────────
model: Optional[FastSpeakerEmbeddingNet] = None

def load_local_model():
    global model
    if HAS_ML_LIBS and MODEL_PATH.exists():
        log.info("Loading model from %s …", MODEL_PATH)
        try:
            m = FastSpeakerEmbeddingNet(embedding_dim=EMBEDDING_DIM)
            state = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            m.load_state_dict(state)
            m.eval()
            model = m
            log.info("Model loaded ✓")
        except Exception as exc:
            log.warning("Could not load saved weights (%s).", exc)

if not os.environ.get("VERCEL"):
    load_local_model()


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve Frontend ───────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

if FRONTEND_DIR.exists():
    # This allows accessing http://127.0.0.1:8000/frontend/index.html
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
else:
    log.warning("Frontend directory not found at %s", FRONTEND_DIR)

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the main frontend page at the root URL."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>SoundSense API</h1><p>Frontend index.html not found.</p>")


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────
class SpeakerSegment(BaseModel):
    speaker: str
    start: float
    end: float
    duration: float

class AnalysisResult(BaseModel):
    file_id: str
    duration_seconds: float
    speech_type: str
    confidence: float
    num_speakers: int
    segments: List[SpeakerSegment]
    message: str


# ─────────────────────────────────────────────
# Audio helpers (Lightweight fallback)
# ─────────────────────────────────────────────
import wave

def get_audio_duration(path: Path) -> float:
    """Get duration of a WAV file without librosa."""
    with wave.open(str(path), 'rb') as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)

def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load WAV, force mono, resample to SAMPLE_RATE."""
    if not HAS_ML_LIBS:
        raise ImportError("librosa not installed. Local audio loading disabled.")
    y, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return y, sr


def extract_mel(segment: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract normalized mel spectrogram — matches notebook exactly."""  # ← fixed: was MFCC
    mel = librosa.feature.melspectrogram(
        y=segment, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel.astype(np.float32)


def segment_audio(y: np.ndarray, sr: int) -> tuple[list, list]:
    """Split audio into fixed-length segments, including the trailing part."""
    segment_len = int(SEGMENT_SEC * sr)
    hop_len     = int(SEGMENT_HOP_SEC * sr)
    segments, timestamps = [], []
    
    for start in range(0, len(y), hop_len):
        end = start + segment_len
        actual_seg = y[start:end]
        
        # Don't process segments shorter than 0.2s (avoid noise at the very end)
        if len(actual_seg) < int(0.2 * sr):
             continue
             
        # Pad with zeros if it's shorter than segment_len (usually the last segment)
        if len(actual_seg) < segment_len:
            actual_seg = np.pad(actual_seg, (0, segment_len - len(actual_seg)))
             
        segments.append(actual_seg)
        timestamps.append((start / sr, min(len(y), end) / sr))
        
    return segments, timestamps


def get_embeddings(segments: list) -> np.ndarray:
    """Run CNN on each segment and return embedding matrix."""  # ← fixed: uses mel, not MFCC
    if not segments:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    mels   = [extract_mel(seg) for seg in segments]
    batch  = torch.tensor(np.stack(mels), dtype=torch.float32)   # (N, mel, time)
    with torch.no_grad():
        embs = model(batch)                                        # (N, 64)
    return embs.cpu().numpy()


def cluster_embeddings(embeddings: np.ndarray, num_speakers: int = 0) -> np.ndarray:
    """Clustering with either a fixed count or Automatic detection via Silhouette Score."""
    if len(embeddings) == 0:
        return np.array([])
    if len(embeddings) < 2:
        return np.zeros(len(embeddings), dtype=int)

    # If user provided a specific number (1 or more)
    if num_speakers > 0:
        n = min(num_speakers, len(embeddings))
        return AgglomerativeClustering(n_clusters=n).fit_predict(embeddings)
    
    # --- AUTO DETECTION MODE (Silhouette Score) ---
    # We evaluate k=2 up to k=6 to find the optimal number of speakers.
    max_k = min(len(embeddings) - 1, 6) 
    if max_k < 2:
         return np.zeros(len(embeddings), dtype=int)

    best_n = 1
    best_score = -1
    
    for k in range(2, max_k + 1):
        try:
            # Using 'complete' linkage can be better at finding small outliers (3rd speaker)
            clustering = AgglomerativeClustering(
                n_clusters=k, 
                metric='cosine', 
                linkage='complete'
            ).fit(embeddings)
            labels = clustering.labels_
            score = silhouette_score(embeddings, labels, metric='cosine')
            
            # Tiered Bonus: Give a significantly higher boost to 3+ speakers 
            # to prevent the model from 'playing it safe' with only 2 clusters.
            bonus = k * 0.10 if k >= 3 else k * 0.02
            adjusted_score = score + bonus
            
            if adjusted_score > best_score:
                best_score = adjusted_score
                actual_silhouette = score
                best_n = k
        except Exception as e:
            log.warning("Clustering failed for k=%d: %s", k, e)
            continue
            
    # Fallback check
    if best_n > 1 and actual_silhouette < 0.01:
        best_n = 1
            
    log.info("Auto-detected %d speakers (Adjusted Score: %.4f)", best_n, best_score)
        
    return AgglomerativeClustering(n_clusters=best_n).fit_predict(embeddings)


def make_speaker_names(labels: np.ndarray) -> list[str]:
    mapping = {old: f"Speaker {i+1}" for i, old in enumerate(sorted(np.unique(labels)))}
    return [mapping[x] for x in labels]


def merge_consecutive_segments(
    timestamps: list[tuple], labels: list[str]
) -> list[dict]:
    """Merge adjacent segments with the same speaker label."""
    if not timestamps:
        return []
    merged = []
    cur_start, cur_end = timestamps[0]
    cur_label = labels[0]
    for i in range(1, len(timestamps)):
        s, e = timestamps[i]
        if labels[i] == cur_label:
            cur_end = e
        else:
            merged.append({"speaker": cur_label, "start": round(cur_start, 2),
                           "end": round(cur_end, 2), "duration": round(cur_end - cur_start, 2)})
            cur_start, cur_end, cur_label = s, e, labels[i]
    merged.append({"speaker": cur_label, "start": round(cur_start, 2),
                   "end": round(cur_end, 2), "duration": round(cur_end - cur_start, 2)})
    return merged


def diarize(y: np.ndarray, sr: int, num_speakers: int = 2) -> list[dict]:
    """Full diarization pipeline — matches notebook flow exactly."""
    segments, timestamps = segment_audio(y, sr)
    if not segments:
        return []
    embeddings   = get_embeddings(segments)
    labels_raw   = cluster_embeddings(embeddings, num_speakers)
    speaker_names = make_speaker_names(labels_raw)
    return merge_consecutive_segments(timestamps, speaker_names)


async def analyze_with_gemini(audio_path: Path) -> Optional[List[dict]]:
    """Use Gemini 1.5 Flash to perform high-precision diarization."""
    # Fetch key fresh from environment (critical for Vercel updates)
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        log.error("Gemini analysis skipped: GEMINI_API_KEY environment variable is empty.")
        return None
        
    try:
        log.info("Connecting to Gemini AI...")
        genai.configure(api_key=api_key)
        model_gemini = genai.GenerativeModel("gemini-1.5-flash")
        
        # Upload file to Gemini
        audio_file = genai.upload_file(path=str(audio_path))
        
        prompt = (
            "You are a professional audio diarization expert. "
            "Analyze this audio and identify the distinct human speakers. "
            "IMPORTANT RULES:\n"
            "1. Ignore background noise, music, or non-speech sounds.\n"
            "2. Do NOT over-count speakers. Only identify a new speaker if the voice is clearly and distinctly different.\n"
            "3. If only one person is talking, output only 'Speaker 1'.\n"
            "4. Provide the start and end times for each segment.\n"
            "Output the result ONLY as a raw JSON list of objects (no markdown blocks):\n"
            "[{\"speaker\": \"Speaker 1\", \"start\": 0.0, \"end\": 2.5}, ...]"
        )
        
        response = model_gemini.generate_content([prompt, audio_file])
        
        # Parse JSON from response
        text = response.text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        
        segments = json.loads(text)
        
        # Clean up Gemini file
        audio_file.delete()
        
        # Format segments
        for s in segments:
            s["duration"] = round(s["end"] - s["start"], 2)
            
        log.info("Gemini verification complete. Detected %d segments.", len(segments))
        return segments
    except Exception as e:
        log.error("Gemini analysis failed: %s", e)
        return None


def count_speakers(segments: list[dict]) -> int:
    return len({s["speaker"] for s in segments})


def infer_speech_type(num_speakers: int, duration: float) -> tuple[str, float]:
    """Heuristic speech-type label since model outputs embeddings, not classes."""
    if num_speakers == 0:
        return "noise_only", 0.90
    if num_speakers == 1:
        return "single_speaker", 0.92
    return "multi_speaker", 0.88


def cleanup_file(path: Path):
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    key = os.getenv("GEMINI_API_KEY")
    return {
        "status": "ok", 
        "gemini_key_detected": key is not None and len(key) > 0,
        "ml_libs_available": HAS_ML_LIBS
    }


@app.post("/analyze", response_model=AnalysisResult)
async def analyze(
    file: UploadFile = File(...),
    num_speakers: int = 0,    # 0 = Auto detect
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    # ── Validate ──────────────────────────────────────────
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(400, "Only .wav files are accepted.")

    file_id   = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{file_id}.wav"

    # ── Save ──────────────────────────────────────────────
    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 50 MB).")
    save_path.write_bytes(contents)
    background_tasks.add_task(cleanup_file, save_path)

    # ── Process ───────────────────────────────────────────
    try:
        if HAS_ML_LIBS:
            y, sr = load_audio(save_path)
            duration = len(y) / sr
        else:
            duration = get_audio_duration(save_path)
    except Exception as exc:
        raise HTTPException(422, f"Could not decode audio: {exc}")

    if duration > MAX_DURATION_SEC:
        raise HTTPException(400, f"Audio too long ({duration:.0f}s). Max {MAX_DURATION_SEC}s.")

    # 1. Try Gemini first if key exists (High Precision)
    gemini_segments = await analyze_with_gemini(save_path)
    
    if gemini_segments:
        segments = gemini_segments
        method = "Gemini 1.5 AI"
    else:
        # 2. Fallback to Local Model (Only if libs available)
        if not HAS_ML_LIBS:
             raise HTTPException(400, "Gemini API key missing and local ML libraries not available on this server.")
        
        segments = diarize(y, sr, num_speakers=num_speakers)
        method = "Local CNN"

    num_speakers = count_speakers(segments)
    speech_type, confidence = infer_speech_type(num_speakers, duration)

    return AnalysisResult(
        file_id          = file_id,
        duration_seconds = round(duration, 2),
        speech_type      = speech_type,
        confidence       = round(confidence, 4),
        num_speakers     = num_speakers,
        segments         = [SpeakerSegment(**s) for s in segments],
        message          = (
            f"Detected {num_speakers} speaker(s) via {method} "
            f"in {duration:.1f}s of {speech_type.replace('_', ' ')} audio."
        ),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)