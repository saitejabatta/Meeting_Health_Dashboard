# post_meeting/pipeline_runner.py
"""Run dashboard-compatible analysis outputs for an assistant recording session."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from assistant.memory import estimate_valence
from assistant.transcriber import TranscriptSegment


@dataclass
class PipelineOutputPaths:
    """Paths to dashboard-compatible CSV outputs produced for a session."""

    meeting_metrics_df: Path
    chunk_level_df: Path
    key_moments_df: Path
    topic_health: Path


def run_existing_pipeline(session_id: str) -> PipelineOutputPaths:
    """
    Build the four CSV files expected by the existing Streamlit dashboard.

    The original repository keeps much of the full analysis pipeline in a
    notebook, not importable Python modules. This runner preserves the same
    dashboard schemas and uses the assistant's saved transcript to skip
    re-transcription when transcript.json exists.
    """
    if load_dotenv is not None:
        load_dotenv()
    session_dir = _recordings_dir() / session_id
    if not session_dir.exists():
        raise FileNotFoundError(f"Assistant session not found: {session_dir}")
    transcript_path = session_dir / "transcript.json"
    audio_path = session_dir / "audio_raw.wav"
    if not transcript_path.exists() and not audio_path.exists():
        raise FileNotFoundError(f"Expected transcript.json or audio_raw.wav in {session_dir}")

    segments = _load_transcript_segments(transcript_path) if transcript_path.exists() else []
    chunk_df = _build_chunk_level_df(session_id, segments)
    metrics_df = _build_meeting_metrics_df(session_id, chunk_df, segments)
    key_moments_df = _build_key_moments_df(chunk_df)
    topic_health_df = _build_topic_health_df(chunk_df)

    pipeline_dir = session_dir / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    paths = PipelineOutputPaths(
        meeting_metrics_df=pipeline_dir / "meeting_metrics_df.csv",
        chunk_level_df=pipeline_dir / "chunk_level_df.csv",
        key_moments_df=pipeline_dir / "key_moments_df.csv",
        topic_health=pipeline_dir / "topic_health.csv",
    )
    _atomic_write_csv(metrics_df, paths.meeting_metrics_df)
    _atomic_write_csv(chunk_df, paths.chunk_level_df)
    _atomic_write_csv(key_moments_df, paths.key_moments_df)
    _atomic_write_csv(topic_health_df, paths.topic_health)
    return paths


def _recordings_dir() -> Path:
    return Path(os.getenv("RECORDINGS_DIR", "./recordings")).expanduser()


def _load_transcript_segments(path: Path) -> list[TranscriptSegment]:
    payload = json.loads(path.read_text())
    segments: list[TranscriptSegment] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        segments.append(
            TranscriptSegment(
                start=float(item.get("start") or 0.0),
                end=float(item.get("end") or item.get("start") or 0.0),
                text=str(item.get("text") or ""),
                speaker_label=str(item.get("speaker_label") or "Speaker_A"),
                emotion_scores=item.get("emotion_scores") if isinstance(item.get("emotion_scores"), dict) else None,
            )
        )
    return sorted(segments, key=lambda segment: (segment.start, segment.end))


def _build_chunk_level_df(session_id: str, segments: list[TranscriptSegment]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        valence = estimate_valence(segment.text)
        pos_score = max(valence, 0.0)
        neg_score = max(-valence, 0.0)
        neu_score = max(0.0, 1.0 - pos_score - neg_score)
        emotions = _emotion_scores(segment, valence)
        rows.append(
            {
                "meeting_id": session_id,
                "chunk_idx": index,
                "chunk_text": segment.text,
                "pos_score": pos_score,
                "neg_score": neg_score,
                "neu_score": neu_score,
                "valence": valence,
                "emo_neutral": emotions["neutral"],
                "emo_surprise": emotions["surprise"],
                "emo_sadness": emotions["sadness"],
                "emo_disgust": emotions["disgust"],
                "emo_anger": emotions["anger"],
                "emo_joy": emotions["joy"],
                "emo_fear": emotions["fear"],
            }
        )
    df = pd.DataFrame(rows, columns=[
        "meeting_id",
        "chunk_idx",
        "chunk_text",
        "pos_score",
        "neg_score",
        "neu_score",
        "valence",
        "emo_neutral",
        "emo_surprise",
        "emo_sadness",
        "emo_disgust",
        "emo_anger",
        "emo_joy",
        "emo_fear",
    ])
    if df.empty:
        df = pd.DataFrame([_empty_chunk_row(session_id)])
    df["delta_valence"] = df["valence"].diff().fillna(0.0)
    df["abs_delta_valence"] = df["delta_valence"].abs()
    return df


def _build_meeting_metrics_df(session_id: str, chunk_df: pd.DataFrame, segments: list[TranscriptSegment]) -> pd.DataFrame:
    avg_valence = float(chunk_df["valence"].mean()) if not chunk_df.empty else 0.0
    health_score = max(0.0, min(100.0, (avg_valence + 1.0) / 2.0 * 100.0))
    tension_score = float((chunk_df["valence"] < -0.2).mean() * 100.0) if not chunk_df.empty else 0.0
    collaboration_score = _collaboration_score(segments)
    label = "healthy" if health_score >= 65 else "watch" if health_score >= 45 else "unhealthy"
    return pd.DataFrame([
        {
            "predicted_health_label": label,
            "health_confidence": 0.65,
            "meeting_health_valence": avg_valence,
            "meeting_health_score": health_score,
            "collaboration_score": collaboration_score,
            "tension_score": tension_score,
            "embedding_model": "assistant_rule_based_fallback",
            "meeting_id": session_id,
            "duration_min": _duration_minutes(segments),
            "word_count": sum(len(segment.text.split()) for segment in segments),
            "questions_per_100_words": _questions_per_100_words(segments),
        }
    ])


def _build_key_moments_df(chunk_df: pd.DataFrame) -> pd.DataFrame:
    if chunk_df.empty:
        return pd.DataFrame(columns=[
            "meeting_id",
            "chunk_idx",
            "chunk_text",
            "pos_score",
            "neg_score",
            "neu_score",
            "valence",
            "delta_valence",
            "abs_delta_valence",
            "shift_rank",
        ])
    moments = chunk_df.sort_values("abs_delta_valence", ascending=False).head(5).copy()
    moments = moments[moments["abs_delta_valence"] > 0]
    if moments.empty:
        moments = chunk_df.head(1).copy()
    moments["shift_rank"] = range(1, len(moments) + 1)
    return moments[[
        "meeting_id",
        "chunk_idx",
        "chunk_text",
        "pos_score",
        "neg_score",
        "neu_score",
        "valence",
        "delta_valence",
        "abs_delta_valence",
        "shift_rank",
    ]]


def _build_topic_health_df(chunk_df: pd.DataFrame) -> pd.DataFrame:
    topic_rows: dict[str, dict[str, Any]] = {}
    for _, row in chunk_df.iterrows():
        topic = _guess_topic(str(row.get("chunk_text", "")))
        if not topic:
            topic = "general"
        data = topic_rows.setdefault(
            topic,
            {
                "topic": topic,
                "topic_id": len(topic_rows),
                "keywords": topic,
                "scores": [],
                "valences": [],
                "meeting_ids": set(),
            },
        )
        valence = float(row.get("valence", 0.0))
        data["scores"].append(max(0.0, min(100.0, (valence + 1.0) / 2.0 * 100.0)))
        data["valences"].append(valence)
        data["meeting_ids"].add(str(row.get("meeting_id", "")))
    rows = []
    for data in topic_rows.values():
        rows.append(
            {
                "topic": data["topic"],
                "topic_id": data["topic_id"],
                "keywords": data["keywords"],
                "avg_health_score": sum(data["scores"]) / len(data["scores"]),
                "avg_meeting_health": sum(data["valences"]) / len(data["valences"]),
                "meeting_count": len(data["meeting_ids"]),
            }
        )
    return pd.DataFrame(rows, columns=["topic", "topic_id", "keywords", "avg_health_score", "avg_meeting_health", "meeting_count"])


def _emotion_scores(segment: TranscriptSegment, valence: float) -> dict[str, float]:
    if segment.emotion_scores:
        return {
            "neutral": float(segment.emotion_scores.get("neutral", 0.0)),
            "surprise": float(segment.emotion_scores.get("surprise", 0.0)),
            "sadness": float(segment.emotion_scores.get("sadness", 0.0)),
            "disgust": float(segment.emotion_scores.get("disgust", 0.0)),
            "anger": float(segment.emotion_scores.get("anger", 0.0)),
            "joy": float(segment.emotion_scores.get("joy", 0.0)),
            "fear": float(segment.emotion_scores.get("fear", 0.0)),
        }
    joy = max(valence, 0.0)
    negative = max(-valence, 0.0)
    return {
        "neutral": max(0.0, 1.0 - joy - negative),
        "surprise": 0.0,
        "sadness": negative * 0.35,
        "disgust": negative * 0.15,
        "anger": negative * 0.25,
        "joy": joy,
        "fear": negative * 0.25,
    }


def _empty_chunk_row(session_id: str) -> dict[str, Any]:
    return {
        "meeting_id": session_id,
        "chunk_idx": 0,
        "chunk_text": "",
        "pos_score": 0.0,
        "neg_score": 0.0,
        "neu_score": 1.0,
        "valence": 0.0,
        "emo_neutral": 1.0,
        "emo_surprise": 0.0,
        "emo_sadness": 0.0,
        "emo_disgust": 0.0,
        "emo_anger": 0.0,
        "emo_joy": 0.0,
        "emo_fear": 0.0,
    }


def _collaboration_score(segments: list[TranscriptSegment]) -> float:
    if not segments:
        return 0.0
    counts: dict[str, int] = {}
    for segment in segments:
        counts[segment.speaker_label] = counts.get(segment.speaker_label, 0) + len(segment.text.split())
    if len(counts) <= 1:
        return 50.0
    total = sum(counts.values()) or 1
    shares = [count / total for count in counts.values()]
    imbalance = max(shares) - min(shares)
    return max(0.0, min(100.0, (1.0 - imbalance) * 100.0))


def _duration_minutes(segments: list[TranscriptSegment]) -> float:
    if not segments:
        return 0.0
    return max(0.0, max(segment.end for segment in segments) - min(segment.start for segment in segments)) / 60.0


def _questions_per_100_words(segments: list[TranscriptSegment]) -> float:
    words = sum(len(segment.text.split()) for segment in segments)
    questions = sum(segment.text.count("?") for segment in segments)
    return questions / max(words, 1) * 100.0


def _guess_topic(text: str) -> str:
    stopwords = {"about", "after", "again", "could", "going", "meeting", "should", "their", "there", "think", "would"}
    tokens = [token for token in re.findall(r"[a-z][a-z0-9-]+", text.lower()) if len(token) > 4 and token not in stopwords]
    return " ".join(tokens[:3])


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".csv", delete=False) as handle:
        temp_path = Path(handle.name)
        df.to_csv(handle, index=False)
    temp_path.replace(path)
