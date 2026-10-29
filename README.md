# Multimodal Attendance & Engagement Agent

Privacy-preserving, **fully offline** classroom intelligence: automatic attendance from video, engagement analysis
from head pose + movement + audio activity, rule-based teaching suggestions and per-student analytics.

## Features
| Epic | Implementation |
|---|---|
| 2. Vision | `src/vision.py`: MediaPipe FaceMesh head pose (OpenCV Haar fallback), centroid tracking, sustained-presence logging |
| 3. Audio | `src/audio.py`: ffmpeg extraction + adaptive voice-activity detection (no transcription) |
| 4. AI | `src/fusion.py` + `src/agent.py`: weighted multimodal fusion, rule-based reasoning, optional local LLM (Ollama) |
| 5. Analytics | Engagement timeline, attention drops, student session analytics, CSV/ZIP export |
| 6. Interface/Deploy | Streamlit UI (`app.py`), Docker, tests |

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# install ffmpeg (optional, needed for audio from video): sudo apt install ffmpeg | brew install ffmpeg
streamlit run app.py
```
Open http://localhost:8501, pick **Demo session** to try without a camera, or upload a classroom video.

## Test
```bash
pip install -r requirements-dev.txt && pytest -q
```

## Deploy
**Docker (any VPS / Render / Railway / Azure / GCP Cloud Run)**
```bash
docker build -t engagement-agent .
docker run -p 8501:8501 engagement-agent        # or: docker compose up -d
```
The container honours `$PORT` for PaaS platforms.

**Streamlit Community Cloud**: push to GitHub, create app with `app.py` as entry. `packages.txt` installs ffmpeg/OpenCV libs.
If MediaPipe fails to install on the target Python version, remove it from `requirements.txt`; the app falls back to OpenCV Haar automatically.

## How it works
1. Video is sampled (default 4 fps), faces detected, tracked with IDs `P01, P02...`.
2. **Attendance** counts a participant only after `min presence` seconds of sustained detection.
3. **Attention** = face roughly facing camera (yaw/pitch thresholds). **Activity** = head movement relative to face width.
4. **Audio** = per-second speech/no-speech from adaptive RMS threshold.
5. **Engagement** (0-100) = weighted average of attention, activity and speech score (weights adjustable in the sidebar; missing signals are re-normalised).
6. The agent turns trends into suggestions: low stretches, attention drops, late fade, long monologues, late arrivals, learners needing a check-in.

## Privacy and ethics
No network calls, no stored frames (only one in-memory preview), no speech content, anonymous IDs, temp files deleted after processing.
Output is decision support; low attention scores are prompts to ask, not judgments. Obtain consent per your institution's policy.

## Limitations and tuning
- Best with a front-facing camera, decent lighting, faces at least ~40 px wide.
- Head pose is a proxy for attention. Tune `yaw_limit`/`pitch_range` in `FaceAnalyzer` for your camera angle.
- IDs can fragment if someone leaves frame for more than 3 s; rename the same person identically in the Attendance table.
- Weights and the `MOVE_THR` constant in `src/fusion.py` are heuristics; calibrate with your own sessions.

## Project links (SkillWallet submission)
Demo: _add your hosted URL_ . GitHub: _add your repo URL_
