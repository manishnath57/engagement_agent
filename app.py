"""Multimodal Attendance & Engagement Agent - Streamlit app (offline, privacy-preserving)."""
import os
import tempfile

import cv2
import pandas as pd
import streamlit as st

from src.agent import generate_suggestions, refine_with_local_llm
from src.audio import extract_audio, speech_activity
from src.demo import make_demo
from src.fusion import build_attendance, build_students, build_timeline, fmt_time
from src.report import build_zip
from src.vision import analyze_video

st.set_page_config(page_title="Attendance & Engagement Agent", page_icon="🎓", layout="wide")
st.title("🎓 Multimodal Attendance & Engagement Agent")
st.caption("Computer vision + audio + agent reasoning. Runs fully offline: no cloud calls, no stored images, no speech transcription.")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Data source", ["Upload video", "Demo session"], horizontal=True)
    window_s = st.slider("Analysis window (s)", 15, 120, 30, 15)
    min_presence = st.slider("Min. presence to count attendance (s)", 3, 120, 10)
    target_fps = st.slider("Sampling rate (frames/s)", 1, 10, 4, help="Lower = faster, less precise")
    st.subheader("Engagement weights")
    w_att = st.slider("Attention (head pose)", 0.0, 1.0, 0.6, 0.05)
    w_act = st.slider("Movement / activity", 0.0, 1.0, 0.2, 0.05)
    w_sp = st.slider("Audio activity", 0.0, 1.0, 0.2, 0.05)
    st.divider()
    use_llm = st.checkbox("Polish suggestions with local LLM (Ollama)", value=False)
    llm_model = st.text_input("Ollama model", "llama3.2", disabled=not use_llm)

weights = {"attention": w_att, "activity": w_act, "speech": w_sp}

# ---------------- Acquire data ----------------
if mode == "Upload video":
    video = st.file_uploader("Classroom video (live-recorded or pre-recorded)", type=["mp4", "mov", "avi", "mkv", "webm"])
    wav = st.file_uploader("Optional separate audio (16-bit WAV)", type=["wav"])
    if st.button("Run analysis", type="primary", disabled=video is None):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            vpath = os.path.join(tmp, "in" + os.path.splitext(video.name)[1])
            with open(vpath, "wb") as f:
                f.write(video.getbuffer())
            bar = st.progress(0.0, text="Analysing video...")
            try:
                vr = analyze_video(vpath, target_fps=target_fps, progress=lambda p: bar.progress(p, text="Analysing video..."))
            except Exception as ex:
                st.error(f"Video analysis failed: {ex}")
                st.stop()
            audio, note = None, ""
            wpath = os.path.join(tmp, "a.wav")
            if wav is not None:
                with open(wpath, "wb") as f:
                    f.write(wav.getbuffer())
                ok = True
            else:
                ok = extract_audio(vpath, wpath)
            if ok:
                try:
                    audio = speech_activity(wpath)
                except Exception as ex:
                    note = f"Audio could not be read ({ex})."
            else:
                note = "No audio track / ffmpeg unavailable: engagement uses vision only."
            bar.empty()
        snap = None if vr.snapshot is None else cv2.cvtColor(vr.snapshot, cv2.COLOR_BGR2RGB)
        st.session_state["res"] = dict(frames=vr.frames, audio=audio, duration=vr.duration, dt=vr.dt,
                                       backend=vr.backend, snapshot=snap, note=note, source=video.name)
else:
    if st.button("Load demo session", type="primary"):
        fr, au, dur, dt = make_demo()
        st.session_state["res"] = dict(frames=fr, audio=au, duration=dur, dt=dt, backend="synthetic demo",
                                       snapshot=None, note="Synthetic data for demonstration.", source="demo")

res = st.session_state.get("res")
if not res:
    st.info("Upload a classroom video (or load the demo session) and click the button to begin.")
    st.stop()

# ---------------- Analytics ----------------
frames, audio, duration, dt = res["frames"], res["audio"], res["duration"], res["dt"]
if frames.empty:
    st.warning("No faces detected. Use a clearer, front-facing camera angle with good lighting.")
    st.stop()

attendance = build_attendance(frames, duration, dt, min_presence_s=float(min_presence))
valid = [int(p[1:]) for p in attendance[attendance.status == "Present"].participant]
timeline = build_timeline(frames, audio, duration, window_s, weights, valid)
students = build_students(frames, duration, dt, window_s, valid)
suggestions = generate_suggestions(timeline, attendance, students, duration, audio, window_s)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Session length", fmt_time(duration))
c2.metric("Present (sustained)", len(valid))
c3.metric("Mean engagement", f"{timeline.engagement.mean():.0f}/100")
c4.metric("Vision backend", res["backend"])
if res.get("note"):
    st.caption("ℹ️ " + res["note"])

t1, t2, t3, t4, t5 = st.tabs(["Attendance", "Engagement", "Students", "Suggestions", "Export"])

with t1:
    st.write("Attendance is marked only after **sustained presence**, not single-frame detection. Edit names to match your roster.")
    shown = attendance.drop(columns=["first_seen_s", "last_seen_s"])
    attendance_edit = st.data_editor(
        shown, hide_index=True, use_container_width=True,
        disabled=[c for c in shown.columns if c != "name"],
        column_config={"present_pct": st.column_config.ProgressColumn("present %", min_value=0, max_value=100, format="%.0f"),
                       "attentive_pct": st.column_config.ProgressColumn("attentive %", min_value=0, max_value=100, format="%.0f")})
    if res.get("snapshot") is not None:
        st.image(res["snapshot"], caption="Preview frame (green = attentive, red = looking away)", use_container_width=True)

with t2:
    chart = timeline.dropna(subset=["engagement"]).set_index("minute")[["engagement", "attention", "activity", "speech"]]
    st.line_chart(chart, y_label="score (0-100)", x_label="minute")
    st.dataframe(timeline[["time", "participants", "attention", "activity", "speech", "engagement", "level"]].round(1),
                 hide_index=True, use_container_width=True)

with t3:
    st.caption("Participants are anonymous IDs. Use these insights to support learners, not to penalise them.")
    st.dataframe(students, hide_index=True, use_container_width=True,
                 column_config={"engagement_score": st.column_config.ProgressColumn("engagement", min_value=0, max_value=100, format="%.0f"),
                                "attentive_pct": st.column_config.ProgressColumn("attentive %", min_value=0, max_value=100, format="%.0f")})

with t4:
    icon = {"High": "🔴", "Medium": "🟠", "Info": "🔵"}
    for s in suggestions:
        with st.container(border=True):
            st.markdown(f"**{icon[s['priority']]} {s['title']}**")
            st.write(s["action"])
            if s["evidence"]:
                st.caption("Evidence: " + s["evidence"])
    if use_llm:
        with st.spinner("Asking local model..."):
            txt = refine_with_local_llm(suggestions, llm_model)
        st.success(txt) if txt else st.warning("Local Ollama server not reachable at localhost:11434.")
    st.caption("The agent is decision support only; the instructor makes the final call.")

with t5:
    meta = {"source": res["source"], "duration": fmt_time(duration), "participants counted": len(valid),
            "mean engagement": f"{timeline.engagement.mean():.0f}/100", "backend": res["backend"]}
    st.download_button("⬇️ Download full report (ZIP)", build_zip(attendance_edit, timeline, students, suggestions, meta),
                       file_name="session_report.zip", mime="application/zip", type="primary")
    st.download_button("⬇️ Attendance CSV", attendance_edit.to_csv(index=False), "attendance.csv", "text/csv")
