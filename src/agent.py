"""Rule-based teaching assistant agent (fully offline). Optional local LLM (Ollama) can polish the wording."""
from __future__ import annotations

import json
import urllib.request

from .fusion import fmt_time


def _s(priority, title, action, evidence):
    return {"priority": priority, "title": title, "action": action, "evidence": evidence}


def _runs(mask):
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def generate_suggestions(timeline, attendance, students, duration, audio=None, window_s=30):
    out = []
    tl = timeline.dropna(subset=["engagement"]).reset_index(drop=True)
    if tl.empty:
        return [_s("Info", "Not enough data", "No participants were detected. Check camera angle and lighting.", "")]
    e = tl.engagement.to_numpy()
    mean_e = float(e.mean())

    if mean_e < 45:
        out.append(_s("High", "Overall engagement is low",
                      "Introduce an interactive element early (poll, think-pair-share, quick quiz).",
                      f"Mean engagement {mean_e:.0f}/100."))

    for a, b in _runs(tl.level.eq("Low").to_numpy()):
        if b - a + 1 >= 2:
            out.append(_s("High", "Sustained low-engagement stretch",
                          "Switch format here: short activity, example, or discussion question.",
                          f"{tl.time[a]} to {fmt_time(tl.start_s[b] + window_s)} stayed in the Low band."))

    drops = []
    for i in range(1, len(e)):
        base = e[max(0, i - 2):i].mean()  # compare with the trailing two windows
        if e[i] - base <= -12 and e[i] < mean_e:
            drops.append((e[i] - base, i))
    for delta, i in sorted(drops)[:3]:
        out.append(_s("Medium", f"Attention drop at {tl.time[i]}",
                      "Review what happened just before: long explanation, slide change or off-topic? Add a check-in question.",
                      f"Engagement fell {abs(delta):.0f} points to {e[i]:.0f}."))

    k = max(1, len(e) // 3)
    first, last = e[:k].mean(), e[-k:].mean()
    if len(e) >= 6 and first - last >= 15:
        out.append(_s("Medium", "Engagement fades late in the session",
                      "Plan a short break or recap about two-thirds through, and put key activities earlier.",
                      f"First third {first:.0f} vs last third {last:.0f}."))
    if len(e) >= 6 and first < 45 <= e[k:].mean():
        out.append(_s("Info", "Slow start",
                      "A warm-up question or quick recap at the start may help learners settle in.",
                      f"First third averaged {first:.0f}."))

    if audio is not None and len(audio):
        share = float(audio.speaking.mean() * 100)
        sm = audio.speaking.rolling(5, center=True, min_periods=1).max().to_numpy() > 0
        longest = max([b - a + 1 for a, b in _runs(sm)] or [0])
        if longest >= 600:
            out.append(_s("Medium", "Long uninterrupted speaking block",
                          "Break long stretches of talk with questions, pauses or learner activities.",
                          f"Longest continuous speech about {longest // 60} min."))
        if share < 5:
            out.append(_s("Info", "Very little speech detected",
                          "Encourage Q&A or discussion, or check microphone placement.",
                          f"Speech present in {share:.0f}% of the session."))

    if len(attendance):
        pres = attendance[attendance.status == "Present"]
        late, early = int(pres.late.sum()), int(pres.left_early.sum())
        if len(pres) and late / len(pres) >= 0.2:
            out.append(_s("Info", "Several late arrivals",
                          "Consider a brief recap for latecomers or a lighter first five minutes.",
                          f"{late} of {len(pres)} participants arrived late."))
        if len(pres) and early / len(pres) >= 0.2:
            out.append(_s("Info", "Several early departures",
                          "Check session length and put key points before the final stretch.",
                          f"{early} of {len(pres)} participants left early."))

    if len(students):
        low = students[students.attentive_pct < 40].participant.tolist()
        if low:
            out.append(_s("Medium", "Some learners may need a check-in",
                          "Offer gentle, private support. Low attention is a prompt to ask, not a conclusion.",
                          "Low attentiveness: " + ", ".join(low[:6]) + (" ..." if len(low) > 6 else "")))

    peak = tl.loc[tl.engagement.idxmax()]
    out.append(_s("Info", f"Strongest moment at {peak.time}",
                  "Note what was happening here and reuse that approach.",
                  f"Engagement {peak.engagement:.0f}/100."))
    order = {"High": 0, "Medium": 1, "Info": 2}
    return sorted(out, key=lambda x: order[x["priority"]])


def refine_with_local_llm(suggestions, model="llama3.2", host="http://localhost:11434", timeout=40):
    """Optional: rewrite suggestions as a short coaching note via a LOCAL Ollama server. Returns None on failure."""
    prompt = ("You are a supportive teaching coach. Rewrite these classroom analytics suggestions as a short, "
              "friendly action plan (max 150 words). Do not invent facts.\n" + json.dumps(suggestions))
    req = urllib.request.Request(host.rstrip("/") + "/api/generate",
                                 data=json.dumps({"model": model, "prompt": prompt, "stream": False}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode()).get("response", "").strip() or None
    except Exception:
        return None
