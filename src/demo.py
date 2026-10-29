"""Synthetic session generator so the app can be demoed/tested without a camera."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_demo(n_students: int = 12, duration: int = 2400, dt: float = 2.0, seed: int = 7):
    rng = np.random.default_rng(seed)

    def p_attn(t):
        dip = 0.65 * np.exp(-((t - 1300) / 150) ** 2)
        fade = 0.2 * max(0.0, (t - 2000) / 400)
        return 0.88 - dip - fade

    rows = []
    for tid in range(1, n_students + 1):
        arrive = float(rng.choice([0, 0, 0, 0, 0, 0, 0, 40, 200, 420])) if tid > 1 else 0.0
        leave = duration - (300 if tid in (5, 9) else 0)
        bias = -0.55 if tid == 3 else rng.normal(0, 0.07)
        for t in np.arange(arrive, leave, dt):
            attentive = int(rng.random() < np.clip(p_attn(t) + bias, 0.02, 0.98))
            move_p = 0.3 if 1100 < t < 1500 else 0.6
            motion = rng.uniform(0.03, 0.12) if rng.random() < move_p else abs(rng.normal(0.008, 0.005))
            rows.append((float(t), tid, attentive, round(float(motion), 4)))
    frames = pd.DataFrame(rows, columns=["t", "track_id", "attentive", "motion"])

    probs = [0.92 if s < 700 else 0.6 if s < 900 else 0.85 if s < 1500 else 0.7 if s < 1600 else 0.5
             for s in range(duration)]
    speaking = (rng.random(duration) < np.array(probs)).astype(int)
    audio = pd.DataFrame({"t": np.arange(duration),
                          "level_db": np.where(speaking == 1, -28, -50) + rng.normal(0, 1.5, duration).round(1),
                          "speaking": speaking})
    return frames, audio, float(duration), float(dt)
