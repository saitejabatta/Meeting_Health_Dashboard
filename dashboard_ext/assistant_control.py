# dashboard_ext/assistant_control.py
"""Streamlit controls for assistant briefing, resume upload, and recordings."""

from __future__ import annotations

import json
import importlib.util
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

from assistant.briefing import parse_briefing_from_text
from assistant.recorder import slugify


def render_assistant_control(recordings_dir: Path) -> None:
    """Render controls for resume upload, proxy briefing, and recording management."""
    st.subheader("Assistant Control")
    st.write("Prepare the assistant to attend a mock interview, use your resume as context, and manage recordings.")

    profile_dir = recordings_dir.parent / "assistant_profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)

    resume_text = _resume_upload_section(profile_dir)
    briefing_text = _briefing_section(profile_dir, resume_text)
    _meeting_join_section(briefing_text)
    _recordings_section(recordings_dir)
    _run_commands_section(briefing_text)


def _resume_upload_section(profile_dir: Path) -> str:
    st.markdown("**1. Upload Resume**")
    uploaded_resume = st.file_uploader("Resume", type=["txt", "md", "pdf", "docx"], key="assistant_resume")
    existing_profiles = sorted(profile_dir.glob("*.json"), reverse=True)
    resume_text = ""

    if uploaded_resume is not None:
        resume_text = _extract_uploaded_text(uploaded_resume)
        if resume_text:
            st.success(f"Loaded {len(resume_text.split())} words from {uploaded_resume.name}.")
            with st.expander("Resume text preview"):
                st.text_area("Extracted resume text", resume_text, height=220)
            if st.button("Save Resume Profile"):
                profile_path = profile_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{slugify(uploaded_resume.name)}.json"
                _atomic_write_json(profile_path, {"filename": uploaded_resume.name, "resume_text": resume_text})
                st.success(f"Saved profile: {profile_path.name}")
        else:
            st.error(_extraction_help(uploaded_resume.name))
    elif existing_profiles:
        selected = st.selectbox("Or load saved resume profile", existing_profiles, format_func=lambda path: path.name)
        payload = json.loads(selected.read_text())
        resume_text = str(payload.get("resume_text", ""))
        with st.expander("Saved resume preview"):
            st.text_area("Resume text", resume_text, height=180)

    manual_resume_text = st.text_area(
        "Resume text override",
        value=resume_text,
        height=180,
        help="Paste resume text here if the uploaded file is scanned or the extractor cannot read it.",
    ).strip()
    if manual_resume_text:
        resume_text = manual_resume_text
        if st.button("Save Pasted Resume Profile"):
            profile_path = profile_dir / f"{datetime.now():%Y%m%d_%H%M%S}_pasted_resume.json"
            _atomic_write_json(profile_path, {"filename": "pasted_resume", "resume_text": resume_text})
            st.success(f"Saved profile: {profile_path.name}")

    return resume_text


def _briefing_section(profile_dir: Path, resume_text: str) -> str:
    st.markdown("**2. Build Meeting Briefing**")
    col_a, col_b = st.columns(2)
    with col_a:
        meeting_title = st.text_input("Meeting title", value="Teams Mock Interview")
        user_name = st.text_input("Your name", value="Sai")
        user_role = st.text_input("Your role", value="candidate")
        mode = st.selectbox("Assistant mode", ["FULL_PROXY", "ADVISOR", "SILENT_OBSERVER", "PARTICIPATOR"], index=0)
    with col_b:
        response_style = st.text_input("Response style", value="concise, confident, technical, honest")
        auto_respond = st.checkbox("Auto respond", value=mode == "FULL_PROXY")
        max_turns = st.number_input("Max speaking turns", min_value=0, max_value=50, value=8 if mode == "FULL_PROXY" else 0)
        display_name = st.text_input("Meeting display name", value=_transparent_display_name(user_name))

    objective = st.text_area(
        "Objective",
        value="Help me answer testing interview questions using my resume and project experience.",
        height=80,
    )
    speaking_persona = st.text_area(
        "Speaking persona",
        value="Sound like me: calm, direct, practical, and specific. Use first person. Avoid overclaiming.",
        height=90,
    )
    decision_authority = st.text_area(
        "Decision authority",
        value="Can answer technical and project-experience questions from my resume. Must not invent experience or make real commitments.",
        height=80,
    )
    escalation_rules = st.text_area(
        "Escalation rules",
        value="Defer salary, visa, legal, confidential company information, and anything not supported by my resume.",
        height=80,
    )
    talking_points = st.text_area(
        "Talking points",
        value="Highlight relevant testing experience; explain projects with metrics; connect dashboard and AI assistant work to QA/testing workflows.",
        height=90,
    )
    topics_to_avoid = st.text_area(
        "Topics to avoid",
        value="Do not exaggerate experience. Do not claim tools, employers, or certifications not present in my resume.",
        height=80,
    )

    briefing_text = _build_briefing_text(
        meeting_title,
        objective,
        user_name,
        user_role,
        mode,
        response_style,
        auto_respond,
        max_turns,
        speaking_persona,
        decision_authority,
        escalation_rules,
        talking_points,
        topics_to_avoid,
        resume_text,
        display_name,
    )
    parsed = parse_briefing_from_text(briefing_text)
    with st.expander("Generated assistant briefing"):
        st.text_area("Copy/paste into CLI if needed", briefing_text, height=320)
        st.json({
            "meeting_title": parsed.meeting_title,
            "agent_mode": parsed.agent_mode.value,
            "user_name": parsed.user_name,
            "user_role": parsed.user_role,
            "speaking_persona": parsed.speaking_persona,
            "decision_authority": parsed.decision_authority,
            "escalation_rules": parsed.escalation_rules,
            "auto_respond": parsed.auto_respond,
            "max_speaking_turns": parsed.max_speaking_turns,
            "assistant_display_name": parsed.assistant_display_name or display_name,
        })

    if st.button("Save Briefing"):
        briefing_path = profile_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{slugify(meeting_title)}_briefing.txt"
        _atomic_write_text(briefing_path, briefing_text)
        st.success(f"Saved briefing: {briefing_path.name}")

    return briefing_text


def _meeting_join_section(briefing_text: str) -> None:
    st.markdown("**3. Join Meeting as Transparent Proxy**")
    platform = st.selectbox("Meeting platform", ["teams", "zoom", "google meet", "local mic", "audio file"])
    meeting_url = st.text_input("Meeting URL", placeholder="Paste Teams, Zoom, or Google Meet link")
    st.info(
        "Use a disclosed display name such as `Sai - AI Assistant`. The assistant can speak on your behalf from the briefing, "
        "but participants should know an assistant is present."
    )

    if platform in {"teams", "zoom", "google meet"}:
        st.code("python3 -m assistant.cli", language="bash")
        if meeting_url:
            st.markdown(
                "Run the CLI, paste the generated briefing, choose "
                f"`{platform}`, and paste this meeting URL when prompted."
            )
        else:
            st.caption("Paste a meeting URL here so this panel becomes a complete launch checklist.")
    elif platform == "audio file":
        st.caption("Use this for the safest first test: upload a recording below, then choose `audio file` in the CLI.")
    else:
        st.caption("Use local mic when you only want the assistant to hear your microphone, not the full meeting audio.")

    if briefing_text:
        st.download_button("Download proxy briefing", briefing_text, file_name="assistant_proxy_briefing.txt")


def _recordings_section(recordings_dir: Path) -> None:
    st.markdown("**4. Recordings and Sessions**")
    recordings_dir.mkdir(parents=True, exist_ok=True)
    uploaded_audio = st.file_uploader(
        "Upload audio recording for file-mode test",
        type=["wav", "mp3", "mp4", "m4a", "webm"],
        key="assistant_audio",
    )
    if uploaded_audio is not None and st.button("Save Uploaded Recording"):
        uploads_dir = recordings_dir / "_uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        upload_name = Path(uploaded_audio.name)
        output_path = uploads_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{slugify(upload_name.stem)}{upload_name.suffix}"
        output_path.write_bytes(uploaded_audio.getbuffer())
        st.success(f"Saved recording: {output_path}")

    sessions = sorted([path for path in recordings_dir.iterdir() if path.is_dir() and not path.name.startswith("_")], reverse=True)
    if sessions:
        selected_session = st.selectbox("Saved assistant sessions", sessions, format_func=lambda path: path.name)
        files = sorted(path.relative_to(selected_session) for path in selected_session.rglob("*") if path.is_file())
        st.write(f"Files in `{selected_session.name}`")
        st.dataframe({"file": [str(path) for path in files]}, use_container_width=True)
    else:
        st.caption("No assistant sessions saved yet.")


def _run_commands_section(briefing_text: str) -> None:
    st.markdown("**5. Run Assistant**")
    st.code("python3 -m assistant.cli", language="bash")
    with st.expander("Suggested Teams mock interview flow"):
        st.markdown(
            "1. Start Teams and join the mock interview.\n"
            "2. Run `python3 -m assistant.cli`.\n"
            "3. Type `paste`, paste the generated briefing, then type `END` on its own line.\n"
            "4. Choose `teams` if testing system audio, or `audio file` for a safer first test.\n"
            "5. Use the disclosed meeting display name shown in the briefing.\n"
            "6. Use `ADVISOR` first if you are checking audio routing, then `FULL_PROXY` for the proxy test."
        )
    if briefing_text:
        st.download_button("Download briefing.txt", briefing_text, file_name="assistant_briefing.txt")


def _build_briefing_text(
    meeting_title: str,
    objective: str,
    user_name: str,
    user_role: str,
    mode: str,
    response_style: str,
    auto_respond: bool,
    max_turns: int,
    speaking_persona: str,
    decision_authority: str,
    escalation_rules: str,
    talking_points: str,
    topics_to_avoid: str,
    resume_text: str,
    display_name: str,
) -> str:
    return f"""Title: {meeting_title}
Objective: {objective}
My name: {user_name}
My role: {user_role}
Assistant display name: {display_name}
Mode: {mode}
Response style: {response_style}
Auto respond: {'yes' if auto_respond else 'no'}
Max speaking turns: {max_turns}
Speaking persona: {speaking_persona}
Decision authority: {decision_authority}
Escalation rules: {escalation_rules}
Talking points: {talking_points}
Avoid: {topics_to_avoid}
Background: Resume and profile context:
{resume_text}
""".strip()


def _transparent_display_name(user_name: str) -> str:
    clean_name = user_name.strip() or "User"
    words = {piece for piece in clean_name.lower().replace("-", " ").split() if piece}
    if "assistant" in words or "ai" in words:
        return clean_name
    return f"{clean_name} - AI Assistant"


def _extract_uploaded_text(uploaded_file: object) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    data = uploaded_file.getvalue()
    if suffix in {".txt", ".md"}:
        return data.decode("utf-8", errors="replace")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except Exception:
            return ""
        temp_path = _write_temp(data, ".pdf")
        try:
            reader = PdfReader(str(temp_path))
            return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        finally:
            temp_path.unlink(missing_ok=True)
    if suffix == ".docx":
        try:
            import docx
        except Exception:
            return ""
        temp_path = _write_temp(data, ".docx")
        try:
            document = docx.Document(str(temp_path))
            return "\n".join(paragraph.text for paragraph in document.paragraphs).strip()
        finally:
            temp_path.unlink(missing_ok=True)
    return ""


def _extraction_help(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf" and importlib.util.find_spec("pypdf") is None:
        return "PDF extraction needs pypdf installed. Run `python3 -m pip install -r requirements_assistant.txt`, or paste the resume text below."
    if suffix == ".docx" and importlib.util.find_spec("docx") is None:
        return "DOCX extraction needs python-docx installed. Run `python3 -m pip install -r requirements_assistant.txt`, or paste the resume text below."
    if suffix == ".pdf":
        return "Could not extract text from this PDF. It may be scanned/image-only. Paste the resume text below."
    return "Could not extract text from that file. Try TXT/MD, or paste the resume text below."


def _write_temp(data: bytes, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(data)
        return Path(handle.name)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2))
