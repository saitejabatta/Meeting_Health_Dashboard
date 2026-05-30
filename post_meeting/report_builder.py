# post_meeting/report_builder.py
"""Build detailed Markdown and JSON reports for assistant meeting sessions."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from assistant.briefing import Briefing, briefing_to_dict
from assistant.memory import RuntimeMemory, runtime_memory_to_dict
from assistant.transcriber import TranscriptSegment
from post_meeting.pipeline_runner import PipelineOutputPaths


@dataclass
class MeetingReport:
    """Saved post-meeting report artifacts and structured report payload."""

    session_id: str
    markdown_path: Path
    json_path: Path
    summary: str
    payload: dict[str, Any]


def build_report(
    session_id: str,
    runtime_memory: RuntimeMemory,
    briefing: Briefing,
    pipeline_outputs: PipelineOutputPaths,
) -> MeetingReport:
    """Build and save a full Markdown and JSON post-meeting report."""
    if load_dotenv is not None:
        load_dotenv()
    session_dir = _recordings_dir() / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    metrics = _read_first_row(pipeline_outputs.meeting_metrics_df)
    chunk_df = pd.read_csv(pipeline_outputs.chunk_level_df) if pipeline_outputs.chunk_level_df.exists() else pd.DataFrame()
    transcript = runtime_memory.full_transcript or _segments_from_chunk_df(chunk_df)
    summary = _executive_summary(runtime_memory, briefing, metrics, transcript)
    follow_up = _suggested_follow_up(runtime_memory, briefing)
    payload = {
        "session_id": session_id,
        "executive_summary": summary,
        "briefing": briefing_to_dict(briefing),
        "metrics": metrics,
        "checklist_results": _load_checklist(session_dir),
        "key_decisions": list(runtime_memory.decisions_detected),
        "action_items": [asdict(item) for item in runtime_memory.action_items_detected],
        "open_questions": _open_questions(runtime_memory),
        "talking_points": {
            "raised": list(runtime_memory.talking_points_raised),
            "pending": list(runtime_memory.talking_points_pending),
        },
        "participation": _participation_analysis(transcript),
        "sentiment": _sentiment_analysis(runtime_memory, metrics),
        "agent_activity": [asdict(item) for item in runtime_memory.agent_responses_given],
        "flags": [asdict(item) for item in runtime_memory.key_moments],
        "full_transcript": [segment.to_dict() for segment in transcript],
        "notable_quotes": _notable_quotes(runtime_memory, transcript),
        "suggested_follow_up_message": follow_up,
    }
    markdown = _render_markdown(payload)
    markdown_path = session_dir / "report.md"
    json_path = session_dir / "report.json"
    _atomic_write_text(markdown_path, markdown)
    _atomic_write_text(json_path, json.dumps(payload, indent=2, default=str))
    return MeetingReport(session_id, markdown_path, json_path, summary, payload)


def _recordings_dir() -> Path:
    return Path(os.getenv("RECORDINGS_DIR", "./recordings")).expanduser()


def _read_first_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def _segments_from_chunk_df(chunk_df: pd.DataFrame) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    for _, row in chunk_df.iterrows():
        index = float(row.get("chunk_idx", len(segments)))
        segments.append(
            TranscriptSegment(
                start=index,
                end=index + 1,
                text=str(row.get("chunk_text", "")),
                speaker_label="Speaker_A",
                emotion_scores={
                    "neutral": float(row.get("emo_neutral", 0.0)),
                    "joy": float(row.get("emo_joy", 0.0)),
                    "anger": float(row.get("emo_anger", 0.0)),
                    "sadness": float(row.get("emo_sadness", 0.0)),
                    "fear": float(row.get("emo_fear", 0.0)),
                    "surprise": float(row.get("emo_surprise", 0.0)),
                    "disgust": float(row.get("emo_disgust", 0.0)),
                },
            )
        )
    return segments


def _executive_summary(
    runtime_memory: RuntimeMemory,
    briefing: Briefing,
    metrics: dict[str, Any],
    transcript: list[TranscriptSegment],
) -> str:
    llm_summary = _llm_summary(runtime_memory, briefing, transcript)
    if llm_summary:
        return llm_summary
    health = metrics.get("meeting_health_score", "N/A")
    decisions = len(runtime_memory.decisions_detected)
    actions = len(runtime_memory.action_items_detected)
    objective = briefing.meeting_objective
    return (
        f"The meeting focused on {objective}. "
        f"The dashboard health score is {health}, with {decisions} decision(s) and {actions} action item(s) captured. "
        "Follow-up should confirm any open questions and missed talking points."
    )


def _llm_summary(runtime_memory: RuntimeMemory, briefing: Briefing, transcript: list[TranscriptSegment]) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    transcript_text = "\n".join(f"{segment.speaker_label}: {segment.text}" for segment in transcript[:80])
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_REPORT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
            messages=[
                {"role": "system", "content": "Write exactly three concise sentences summarizing a meeting."},
                {
                    "role": "user",
                    "content": (
                        f"Objective: {briefing.meeting_objective}\n"
                        f"Decisions: {runtime_memory.decisions_detected}\n"
                        f"Actions: {[asdict(item) for item in runtime_memory.action_items_detected]}\n"
                        f"Transcript:\n{transcript_text[:12000]}"
                    ),
                },
            ],
            temperature=0.2,
            timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "20")),
        )
        return (response.choices[0].message.content or "").strip() or None
    except Exception:
        return None


def _load_checklist(session_dir: Path) -> list[dict[str, Any]]:
    metadata_path = session_dir / "session_metadata.json"
    if not metadata_path.exists():
        return []
    metadata = json.loads(metadata_path.read_text())
    checklist = metadata.get("checklist") or {}
    return checklist.get("items", []) if isinstance(checklist, dict) else []


def _open_questions(runtime_memory: RuntimeMemory) -> list[str]:
    return [question for question in runtime_memory.questions_raised if question not in runtime_memory.questions_answered]


def _participation_analysis(transcript: list[TranscriptSegment]) -> dict[str, Any]:
    talk_words: dict[str, int] = {}
    for segment in transcript:
        talk_words[segment.speaker_label] = talk_words.get(segment.speaker_label, 0) + len(segment.text.split())
    total = sum(talk_words.values()) or 1
    shares = {speaker: words / total for speaker, words in talk_words.items()}
    equity_score = 100.0
    if shares:
        equity_score = max(0.0, min(100.0, (1.0 - (max(shares.values()) - min(shares.values()))) * 100.0))
    return {
        "talk_words": talk_words,
        "talk_share": shares,
        "equity_score": equity_score,
        "spoke_most": max(talk_words, key=talk_words.get) if talk_words else None,
        "spoke_least": min(talk_words, key=talk_words.get) if talk_words else None,
    }


def _sentiment_analysis(runtime_memory: RuntimeMemory, metrics: dict[str, Any]) -> dict[str, Any]:
    timeline = list(runtime_memory.sentiment_timeline)
    if timeline:
        best = max(timeline, key=lambda item: item.get("valence", 0.0))
        worst = min(timeline, key=lambda item: item.get("valence", 0.0))
    else:
        best = {}
        worst = {}
    return {
        "timeline": timeline,
        "health_score": metrics.get("meeting_health_score"),
        "collaboration_score": metrics.get("collaboration_score"),
        "tension_score": metrics.get("tension_score"),
        "best_mood_point": best,
        "highest_tension_point": worst,
    }


def _notable_quotes(runtime_memory: RuntimeMemory, transcript: list[TranscriptSegment]) -> list[str]:
    quotes = [moment.text for moment in runtime_memory.key_moments if moment.type in {"decision", "key_point"}]
    for segment in transcript:
        if len(quotes) >= 5:
            break
        if len(segment.text.split()) >= 12 and segment.text not in quotes:
            quotes.append(segment.text)
    return quotes[:5]


def _suggested_follow_up(runtime_memory: RuntimeMemory, briefing: Briefing) -> str:
    llm_message = _llm_follow_up(runtime_memory, briefing)
    if llm_message:
        return llm_message
    actions = "; ".join(item.what for item in runtime_memory.action_items_detected) or "confirm next steps"
    questions = "; ".join(_open_questions(runtime_memory)) or "none"
    pending = "; ".join(runtime_memory.talking_points_pending) or "none"
    return (
        f"Hi everyone, thanks for the discussion. My notes show these next steps: {actions}. "
        f"Open questions: {questions}. Items to follow up separately: {pending}."
    )


def _llm_follow_up(runtime_memory: RuntimeMemory, briefing: Briefing) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_REPORT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
            messages=[
                {"role": "system", "content": f"Draft a concise {briefing.response_style} follow-up message."},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "decisions": runtime_memory.decisions_detected,
                            "action_items": [asdict(item) for item in runtime_memory.action_items_detected],
                            "open_questions": _open_questions(runtime_memory),
                            "pending_talking_points": runtime_memory.talking_points_pending,
                        }
                    ),
                },
            ],
            temperature=0.3,
            timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "20")),
        )
        return (response.choices[0].message.content or "").strip() or None
    except Exception:
        return None


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Meeting Report: {payload['session_id']}",
        "",
        "## 1. Executive Summary",
        payload["executive_summary"],
        "",
        "## 2. Checklist Results",
    ]
    checklist = payload["checklist_results"]
    if checklist:
        lines.extend(["| Item | Status | Raised | Resolution |", "| --- | --- | --- | --- |"])
        for item in checklist:
            warning = " **WARNING**" if item.get("status") == "missed" else ""
            lines.append(
                f"| {item.get('description', '')} | {item.get('status', '')}{warning} | "
                f"{item.get('raised_at') or ''} | {item.get('resolution_summary') or item.get('suggested_follow_up') or ''} |"
            )
    else:
        lines.append("No checklist items were stored.")
    lines.extend(["", "## 3. Key Decisions"])
    lines.extend(_numbered(payload["key_decisions"]))
    lines.extend(["", "## 4. Action Items"])
    action_items = payload["action_items"]
    if action_items:
        lines.extend(["| What | Who | By When | Confidence | Source |", "| --- | --- | --- | --- | --- |"])
        for item in action_items:
            lines.append(
                f"| {item.get('what', '')} | {item.get('who', '')} | {item.get('by_when') or ''} | "
                f"{item.get('confidence', '')} | {item.get('source', '')} |"
            )
    else:
        lines.append("No action items detected.")
    lines.extend(["", "## 5. Open Questions"])
    lines.extend(_bullets(payload["open_questions"]))
    lines.extend(["", "## 6. Talking Points Coverage"])
    raised = payload["talking_points"]["raised"]
    pending = payload["talking_points"]["pending"]
    total = len(raised) + len(pending)
    lines.append(f"{len(raised)} / {total} raised")
    lines.extend(_bullets([f"Raised: {item}" for item in raised] + [f"Not raised: {item}" for item in pending]))
    lines.extend(["", "## 7. Participation Analysis"])
    participation = payload["participation"]
    lines.append(f"Equity score: {participation.get('equity_score', 0):.1f}")
    lines.append(f"Spoke most: {participation.get('spoke_most')}")
    lines.append(f"Spoke least: {participation.get('spoke_least')}")
    lines.extend(["", "## 8. Sentiment & Health"])
    sentiment = payload["sentiment"]
    lines.append(f"Health: {sentiment.get('health_score')}")
    lines.append(f"Collaboration: {sentiment.get('collaboration_score')}")
    lines.append(f"Tension: {sentiment.get('tension_score')}")
    lines.extend(["", "## 9. Agent Activity"])
    lines.extend(_bullets([f"{item.get('trigger')}: {item.get('text')}" for item in payload["agent_activity"]]))
    lines.extend(["", "## 10. Full Transcript"])
    for segment in payload["full_transcript"]:
        lines.append(f"- [{_format_seconds(segment.get('start', 0))}] {segment.get('speaker_label')}: {segment.get('text')}")
    lines.extend(["", "## 11. Notable Quotes"])
    lines.extend(_bullets(payload["notable_quotes"]))
    lines.extend(["", "## 12. Suggested Follow-Up Message", payload["suggested_follow_up_message"], ""])
    return "\n".join(lines)


def _numbered(items: list[str]) -> list[str]:
    return [f"{index}. {item}" for index, item in enumerate(items, start=1)] or ["None captured."]


def _bullets(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items] or ["- None captured."]


def _format_seconds(value: object) -> str:
    seconds = int(float(value or 0))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    temp_path.replace(path)
