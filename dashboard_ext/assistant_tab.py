# dashboard_ext/assistant_tab.py
"""Streamlit Assistant Session tab for meeting recordings and agent activity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


def render_assistant_tab(session_dir: Path | None, selected_session: str | None) -> None:
    """Render assistant session metadata, checklist, activity, audio, and transcript."""
    st.subheader("Assistant Session")
    if session_dir is None or not selected_session:
        st.info("Select an assistant session in the sidebar to view assistant-specific details.")
        return
    metadata_path = session_dir / "session_metadata.json"
    if not metadata_path.exists():
        st.info("No assistant metadata found for this session yet.")
        return
    metadata = json.loads(metadata_path.read_text())
    briefing = metadata.get("briefing", {})
    overview_cols = st.columns(5)
    overview_cols[0].metric("Title", str(briefing.get("meeting_title", selected_session)))
    overview_cols[1].metric("Duration", f"{float(metadata.get('duration_seconds', 0.0)) / 60:.1f} min")
    overview_cols[2].metric("Agent Mode", str(briefing.get("agent_mode", "N/A")).replace("_", " ").title())
    overview_cols[3].metric("Speakers", int(metadata.get("speaker_count", 0)))
    overview_cols[4].metric("Words", int(metadata.get("total_words_spoken", 0)))

    st.markdown("**Checklist Completion**")
    checklist = (metadata.get("checklist") or {}).get("items", [])
    if checklist:
        checklist_df = pd.DataFrame(checklist)
        st.dataframe(checklist_df, use_container_width=True, hide_index=True)
        missed = checklist_df[checklist_df["status"] == "missed"] if "status" in checklist_df else pd.DataFrame()
        for _, row in missed.iterrows():
            st.error(f"Missed: {row.get('description')} - {row.get('suggested_follow_up')}")
    else:
        st.caption("No checklist items were stored.")

    st.markdown("**Agent Activity Log**")
    responses = (metadata.get("agent_responses_given") or {}).get("log", [])
    if responses:
        for response in responses:
            with st.expander(f"{_format_seconds(response.get('timestamp'))} - {response.get('trigger', 'Response')}"):
                st.write(response.get("text", ""))
    else:
        st.caption("No agent responses were logged.")

    st.markdown("**Key Moments Timeline**")
    runtime = metadata.get("runtime_memory", {})
    key_moments = runtime.get("key_moments", [])
    if key_moments:
        moments_df = pd.DataFrame(key_moments)
        try:
            import plotly.express as px

            fig = px.scatter(
                moments_df,
                x="timestamp",
                y="type",
                color="type",
                hover_data=["message", "text", "importance"],
                title="Key Moments",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.scatter_chart(moments_df, x="timestamp", y="type", color="type")
        st.dataframe(moments_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No key moments were logged.")

    st.markdown("**Talking Points Tracker**")
    status = metadata.get("talking_points_status", {})
    raised = status.get("raised", [])
    pending = status.get("pending", [])
    total = len(raised) + len(pending)
    st.progress((len(raised) / total) if total else 1.0, text=f"{len(raised)} of {total} talking points raised")
    for point in raised:
        st.success(f"Raised: {point}")
    for point in pending:
        st.warning(f"Pending: {point}")

    audio_path = session_dir / "audio.mp3"
    if audio_path.exists():
        st.markdown("**Audio Player**")
        st.audio(str(audio_path))

    st.markdown("**Full Transcript**")
    transcript = _load_transcript(session_dir)
    if transcript:
        for segment in transcript:
            label = segment.get("speaker_label", "Speaker")
            text = segment.get("text", "")
            timestamp = _format_seconds(segment.get("start"))
            with st.expander(f"{timestamp} {label}: {text[:90]}"):
                st.write(text)
                emotion_scores = segment.get("emotion_scores")
                if emotion_scores:
                    st.json(emotion_scores)
    else:
        st.caption("No transcript file found.")


def _load_transcript(session_dir: Path) -> list[dict[str, Any]]:
    transcript_path = session_dir / "transcript.json"
    if not transcript_path.exists():
        return []
    payload = json.loads(transcript_path.read_text())
    return payload if isinstance(payload, list) else []


def _format_seconds(value: object) -> str:
    try:
        seconds = int(float(value))
    except (TypeError, ValueError):
        seconds = 0
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
