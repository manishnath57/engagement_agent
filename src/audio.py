"""Audio module: offline voice-activity estimation (no speech content is recorded or transcribed)."""
from __future__ import annotations

import shutil
import subprocess
import wave

import numpy as np
import pandas as pd


def extract_audio(video_path: str, out_wav: str) -> bool:
    """Extract 16 kHz mono PCM using ffmpeg. Returns False if ffmpeg is missing or the file has no audio."""
    if shutil.which("ffmpeg") is None:
        return False
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000",
           "-acodec", "pcm_s16le", out_wav]
    try:
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900)
    except Exception:
        return False
    return r.returncode == 0


def _read_wav(path: str):
    with wave.open(path, "rb") as wf:
        sr, ch, sw = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw != 2:
        raise ValueError("Only 16-bit PCM WAV is supported")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def activity_from_samples(x: np.ndarray, sr: int, margin_db: float = 8.0, floor_db: float = -55.0) -> pd.DataFrame:
    """Per-second RMS level and adaptive speech/no-speech decision."""
    n = int(len(x) // sr)
    if n == 0:
        return pd.DataFrame(columns=["t", "level_db", "speaking"])
    seg = x[: n * sr].reshape(n, sr)
    rms = np.sqrt((seg ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    noise = np.percentile(db, 20)
    speaking = ((db > noise + margin_db) & (db > floor_db)).astype(int)
    return pd.DataFrame({"t": np.arange(n), "level_db": db.round(1), "speaking": speaking})


def speech_activity(wav_path: str) -> pd.DataFrame:
    x, sr = _read_wav(wav_path)
    return activity_from_samples(x, sr)
