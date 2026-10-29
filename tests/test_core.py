import numpy as np
import pandas as pd

from src.agent import generate_suggestions
from src.audio import activity_from_samples
from src.demo import make_demo
from src.fusion import build_attendance, build_students, build_timeline
from src.report import build_zip
from src.vision import CentroidTracker, Detection


def _pipeline(window=30):
    frames, audio, dur, dt = make_demo()
    att = build_attendance(frames, dur, dt)
    valid = [int(p[1:]) for p in att[att.status == "Present"].participant]
    tl = build_timeline(frames, audio, dur, window, valid_ids=valid)
    st = build_students(frames, dur, dt, window, valid_ids=valid)
    return frames, audio, dur, dt, att, tl, st


def test_attendance_sustained_presence():
    frames = pd.DataFrame({"t": [0, 1, 2, 3, 0], "track_id": [1, 1, 1, 1, 2],
                           "attentive": 1, "motion": 0.0})
    att = build_attendance(frames, 100, 1.0, min_presence_s=3)
    assert att.set_index("participant").status.to_dict() == {"P01": "Present", "P02": "Not counted (brief)"}


def test_timeline_and_suggestions():
    _, audio, dur, dt, att, tl, st = _pipeline()
    assert tl.engagement.between(0, 100).all()
    sug = generate_suggestions(tl, att, st, dur, audio)
    titles = " ".join(s["title"] for s in sug)
    assert "low-engagement" in titles or "drop" in titles
    assert "speaking block" in titles
    assert any("check-in" in s["title"] for s in sug)  # participant P03 is low by design


def test_audio_vad():
    sr = 16000
    quiet = np.random.default_rng(0).normal(0, 0.001, sr * 5)
    loud = np.random.default_rng(1).normal(0, 0.1, sr * 5)
    df = activity_from_samples(np.concatenate([quiet, loud]).astype(np.float32), sr)
    assert df.speaking[:5].sum() == 0 and df.speaking[5:].sum() == 5


def test_tracker_keeps_id():
    tr = CentroidTracker()
    a = tr.update([Detection(100, 100, 50, 50, True)], 0.0)[0][0]
    b = tr.update([Detection(105, 102, 50, 50, True)], 0.25)[0][0]
    assert a == b


def test_export_zip():
    _, _, dur, dt, att, tl, st = _pipeline()
    z = build_zip(att, tl, st, generate_suggestions(tl, att, st, dur), {"duration": dur})
    assert len(z) > 500
