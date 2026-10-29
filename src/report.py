"""Exports: CSVs + Markdown summary bundled as a ZIP."""
from __future__ import annotations

import io
import zipfile
from datetime import datetime


def suggestions_markdown(suggestions, meta: dict) -> str:
    lines = ["# Session Report", "", f"Generated: {datetime.now():%Y-%m-%d %H:%M}"]
    lines += [f"- **{k}**: {v}" for k, v in meta.items()]
    lines += ["", "## Teaching suggestions", ""]
    for s in suggestions:
        lines += [f"### [{s['priority']}] {s['title']}", s["action"], f"_Evidence: {s['evidence']}_", ""]
    lines.append("_Decision-support only. Participants are anonymous IDs; no images or speech content are stored._")
    return "\n".join(lines)


def build_zip(attendance, timeline, students, suggestions, meta) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("attendance.csv", attendance.drop(columns=["first_seen_s", "last_seen_s"], errors="ignore").to_csv(index=False))
        z.writestr("engagement_timeline.csv", timeline.to_csv(index=False))
        z.writestr("student_analytics.csv", students.to_csv(index=False))
        z.writestr("report.md", suggestions_markdown(suggestions, meta))
    return buf.getvalue()
