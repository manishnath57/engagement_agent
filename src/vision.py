"""Vision module: face detection, head-pose attention, tracking and sustained-presence logging.

Backends: MediaPipe FaceMesh (head pose, preferred) -> OpenCV Haar cascades (fallback).
All processing is local; frames are never stored except one annotated preview snapshot in memory.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd

try:
    import mediapipe as mp
    _HAS_MP = hasattr(mp, "solutions")
except Exception:  # pragma: no cover
    mp = None
    _HAS_MP = False


@dataclass
class Detection:
    cx: float
    cy: float
    w: float
    h: float
    attentive: bool


class FaceAnalyzer:
    def __init__(self, max_faces: int = 40, yaw_limit: float = 0.16, pitch_range=(0.40, 0.74)):
        self.yaw_limit, self.pitch_range = yaw_limit, pitch_range
        self.backend = "mediapipe" if _HAS_MP else "opencv-haar"
        if _HAS_MP:
            self.mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=max_faces,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
        else:
            if not hasattr(cv2, "CascadeClassifier"):
                raise RuntimeError(
                    "Your OpenCV version has no face detector. "
                    "Run: pip install \"opencv-python-headless<5\"")
            base = cv2.data.haarcascades
            self.front = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
            self.profile = cv2.CascadeClassifier(base + "haarcascade_profileface.xml")

    def detect(self, frame_bgr) -> list[Detection]:
        return self._mp(frame_bgr) if _HAS_MP else self._haar(frame_bgr)

    def _mp(self, frame):
        h, w = frame.shape[:2]
        res = self.mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        out = []
        for fl in res.multi_face_landmarks or []:
            lm = fl.landmark
            xs = [p.x for p in lm]
            ys = [p.y for p in lm]
            x0, x1, y0, y1 = min(xs) * w, max(xs) * w, min(ys) * h, max(ys) * h
            fw = max(lm[454].x - lm[234].x, 1e-6)
            fh = max(lm[152].y - lm[10].y, 1e-6)
            yaw = (lm[1].x - lm[234].x) / fw - 0.5        # 0 = facing camera
            pitch = (lm[1].y - lm[10].y) / fh              # ~0.55 neutral; higher = looking down
            att = abs(yaw) <= self.yaw_limit and self.pitch_range[0] <= pitch <= self.pitch_range[1]
            out.append(Detection((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, bool(att)))
        return out

    def _haar(self, frame):
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        fr = self.front.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        pr = self.profile.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        out = [Detection(x + w / 2, y + h / 2, w, h, True) for x, y, w, h in fr]
        for x, y, w, h in pr:  # profile faces = looking sideways -> not attentive
            cx, cy = x + w / 2, y + h / 2
            if not any(abs(cx - d.cx) < d.w / 2 and abs(cy - d.cy) < d.h / 2 for d in out):
                out.append(Detection(cx, cy, w, h, False))
        return out


class CentroidTracker:
    """Greedy nearest-centroid tracker; a track survives `max_missing_s` without detections."""

    def __init__(self, dist_factor: float = 1.5, max_missing_s: float = 3.0):
        self.dist_factor, self.max_missing_s = dist_factor, max_missing_s
        self.tracks: dict[int, dict] = {}
        self.next_id = 1

    def update(self, dets: list[Detection], t: float):
        pairs = []
        for tid, tr in self.tracks.items():
            for j, d in enumerate(dets):
                dist = math.hypot(tr["cx"] - d.cx, tr["cy"] - d.cy)
                if dist <= self.dist_factor * max(tr["w"], d.w):
                    pairs.append((dist, tid, j))
        pairs.sort()
        used_t, used_d, result = set(), set(), []
        for dist, tid, j in pairs:
            if tid in used_t or j in used_d:
                continue
            d = dets[j]
            motion = dist / max(d.w, 1.0)
            self.tracks[tid].update(cx=d.cx, cy=d.cy, w=d.w, last_t=t)
            used_t.add(tid)
            used_d.add(j)
            result.append((tid, d, motion))
        for j, d in enumerate(dets):
            if j not in used_d:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = dict(cx=d.cx, cy=d.cy, w=d.w, last_t=t)
                result.append((tid, d, 0.0))
        for tid in [k for k, v in self.tracks.items() if t - v["last_t"] > self.max_missing_s]:
            del self.tracks[tid]
        return result


@dataclass
class VisionResult:
    frames: pd.DataFrame
    duration: float
    dt: float
    backend: str
    snapshot: Optional[np.ndarray]


def analyze_video(path: str, target_fps: float = 4.0, max_width: int = 960,
                  progress: Optional[Callable[[float], None]] = None) -> VisionResult:
    analyzer, tracker = FaceAnalyzer(), CentroidTracker()  # created first: fail before the file is opened
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / target_fps)))
    rows, best_n, snapshot, idx, last_t = [], -1, None, 0, 0.0
    try:
        while True:
            if not cap.grab():
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    t = idx / fps
                    last_t = t
                    h, w = frame.shape[:2]
                    if w > max_width:
                        frame = cv2.resize(frame, (max_width, int(h * max_width / w)))
                    dets = analyzer.detect(frame)
                    tracked = tracker.update(dets, t)
                    for tid, d, motion in tracked:
                        rows.append((round(t, 3), tid, int(d.attentive), round(motion, 4)))
                    if len(tracked) > best_n:
                        best_n = len(tracked)
                        snap = frame.copy()
                        for tid, d, _ in tracked:
                            col = (80, 200, 80) if d.attentive else (60, 60, 230)
                            p1 = (int(d.cx - d.w / 2), int(d.cy - d.h / 2))
                            p2 = (int(d.cx + d.w / 2), int(d.cy + d.h / 2))
                            cv2.rectangle(snap, p1, p2, col, 2)
                            cv2.putText(snap, f"P{tid:02d}", (p1[0], max(12, p1[1] - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
                        snapshot = snap
            idx += 1
            if progress and total and idx % (step * 10) == 0:
                progress(min(idx / total, 1.0))
    finally:
        cap.release()  # always release, so Windows can delete the temp file
    if progress:
        progress(1.0)
    duration = (total / fps) if total else last_t
    df = pd.DataFrame(rows, columns=["t", "track_id", "attentive", "motion"])
    return VisionResult(df, float(duration), step / fps, analyzer.backend, snapshot)