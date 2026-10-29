"""Multimodal fusion: attendance, engagement timeline and per-student analytics.

Input `frames` DataFrame columns: t (s), track_id, attentive (0/1), motion (face-widths moved per sample).
Input `audio` DataFrame columns: t (int s), level_db, speaking (0/1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MOVE_THR = 0.02  # motion above this (fraction of face width per sample) counts as "moving"
DEFAULT_WEIGHTS = {"attention": 0.6, "activity": 0.2, "speech": 0.2}


def fmt_time(s: float) -> str:
    s = int(round(float(s)))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def pid(track_id: int) -> str:
    return f"P{int(track_id):02d}"


def longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def build_attendance(frames, duration, dt, min_presence_s=10.0, late_frac=0.15, early_frac=0.10):
    cols = ["participant", "name", "status", "first_seen", "last_seen", "present_s",
            "present_pct", "attentive_pct", "activity_pct", "late", "left_early",
            "first_seen_s", "last_seen_s"]
    if frames is None or frames.empty:
        return pd.DataFrame(columns=cols)
    g = frames.groupby("track_id")
    a = g.agg(first=("t", "min"), last=("t", "max"), n=("t", "size"),
              attentive=("attentive", "mean"),
              activity=("motion", lambda m: float((m > MOVE_THR).mean()))).reset_index()
    out = pd.DataFrame({
        "participant": a.track_id.map(pid),
        "name": a.track_id.map(pid),
        "present_s": (a.n * dt).round(1),
        "first_seen_s": a["first"].round(1),
        "last_seen_s": a["last"].round(1),
        "attentive_pct": (a.attentive * 100).round(1),
        "activity_pct": (a.activity * 100).round(1),
    })
    out["present_pct"] = (out.present_s / max(duration, 1e-9) * 100).clip(upper=100).round(1)
    out["status"] = np.where(out.present_s >= min_presence_s, "Present", "Not counted (brief)")
    out["first_seen"] = out.first_seen_s.map(fmt_time)
    out["last_seen"] = out.last_seen_s.map(fmt_time)
    out["late"] = out.first_seen_s > late_frac * duration
    out["left_early"] = out.last_seen_s < (1 - early_frac) * duration
    return out[cols].sort_values("participant").reset_index(drop=True)


def _fuse(att, act, sp, w):
    if np.isnan(att):
        return np.nan
    parts = [(att, w["attention"])]
    if not np.isnan(act):
        parts.append((act, w["activity"]))
    if not np.isnan(sp):
        parts.append((min(sp / 40.0, 1.0) * 100.0, w["speech"]))  # 40% speech time saturates the audio score
    tot = sum(x for _, x in parts)
    return sum(v * x for v, x in parts) / tot if tot else np.nan


def level_of(e):
    if np.isnan(e):
        return "No data"
    return "High" if e >= 70 else "Medium" if e >= 45 else "Low"


def build_timeline(frames, audio, duration, window_s=30, weights=None, valid_ids=None):
    w = dict(DEFAULT_WEIGHTS)
    w.update(weights or {})
    f = frames if valid_ids is None else frames[frames.track_id.isin(valid_ids)]
    has_audio = audio is not None and len(audio) > 0
    rows = []
    for s in np.arange(0, max(duration, 1e-9), window_s):
        e = s + window_s
        fw = f[(f.t >= s) & (f.t < e)]
        if fw.empty:
            att = act = np.nan
            n = 0
        else:
            att = fw.attentive.mean() * 100
            act = (fw.motion > MOVE_THR).mean() * 100
            n = fw.track_id.nunique()
        sp = np.nan
        if has_audio:
            aw = audio[(audio.t >= s) & (audio.t < e)]
            if len(aw):
                sp = aw.speaking.mean() * 100
        eng = _fuse(att, act, sp, w)
        rows.append({"start_s": float(s), "time": fmt_time(s), "minute": round(s / 60, 2),
                     "participants": n, "attention": att, "activity": act, "speech": sp,
                     "engagement": eng, "level": level_of(eng)})
    return pd.DataFrame(rows)


def build_students(frames, duration, dt, window_s=30, valid_ids=None):
    cols = ["participant", "attentive_pct", "activity_pct", "longest_stillness_s",
            "low_attention_windows", "engagement_score"]
    f = frames if valid_ids is None else frames[frames.track_id.isin(valid_ids)]
    if f.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for tid, grp in f.groupby("track_id"):
        grp = grp.sort_values("t")
        att = grp.attentive.mean() * 100
        act = (grp.motion > MOVE_THR).mean() * 100
        still = (grp.motion <= MOVE_THR).to_numpy()
        wa = grp.groupby((grp.t // window_s).astype(int)).attentive.mean()
        rows.append({"participant": pid(tid), "attentive_pct": round(att, 1),
                     "activity_pct": round(act, 1),
                     "longest_stillness_s": round(longest_run(still) * dt, 1),
                     "low_attention_windows": int((wa < 0.4).sum()),
                     "engagement_score": round(0.7 * att + 0.3 * act, 1)})
    return pd.DataFrame(rows, columns=cols).sort_values("engagement_score").reset_index(drop=True)
