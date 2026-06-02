# assistant/session.py
"""Meeting session orchestration and live checklist tracking."""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
import wave
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from assistant.agent import MeetingAgent
from assistant.audio_capture import AudioCapture
from assistant.briefing import Briefing, AgentMode
from assistant.memory import RuntimeMemory, learn_from_briefing
from assistant.recorder import Recorder, generate_session_id
from assistant.responder import Responder
from assistant.speaker_tracker import SpeakerTracker, assign_speakers_to_segments
from assistant.transcriber import LiveTranscriber, TranscriptSegment, normalize_audio_file_to_wav

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if TYPE_CHECKING:
    from post_meeting.pipeline_runner import PipelineOutputPaths

ChecklistStatus = Literal["pending", "raised", "resolved", "missed"]
JoinSource = Literal["mic", "system", "both"]


@dataclass
class SessionReport:
    """Summary of saved artifacts and processing outputs for one session."""

    session_id: str
    session_dir: Path
    audio_mp3_path: Path | None
    transcript_paths: dict[str, Path]
    metadata_path: Path
    pipeline_outputs: "PipelineOutputPaths"
    duration_seconds: float
    speaker_count: int
    word_count: int
    live_summary: str


@dataclass
class ChecklistItem:
    """A single do-not-miss item tracked during a meeting."""

    description: str
    source: Literal["talking_point", "objective", "user_defined"]
    status: ChecklistStatus = "pending"
    raised_at: float | None = None
    resolved_at: float | None = None
    resolution_summary: str | None = None
    suggested_follow_up: str | None = None


@dataclass
class DoNotMissChecklist:
    """Tracks briefing-derived items that must be covered before the meeting ends."""

    items: list[ChecklistItem] = field(default_factory=list)

    @classmethod
    def from_briefing(cls, briefing: Briefing) -> "DoNotMissChecklist":
        """Build a checklist from the meeting objective, talking points, and custom instructions."""
        items: list[ChecklistItem] = []
        if briefing.meeting_objective.strip():
            items.append(ChecklistItem(briefing.meeting_objective.strip(), "objective"))
        for point in briefing.talking_points:
            items.append(ChecklistItem(point.strip(), "talking_point"))
        for instruction in briefing.custom_instructions:
            if _looks_like_checklist_instruction(instruction):
                items.append(ChecklistItem(instruction.strip(), "user_defined"))
        return cls([item for item in items if item.description])

    def update_from_segment(self, segment: TranscriptSegment) -> None:
        """Update checklist statuses based on a new transcript segment."""
        for item in self.items:
            if item.status in {"resolved", "missed"}:
                continue
            similarity = _semanticish_match(item.description, segment.text)
            if similarity >= 0.62 and item.status == "pending":
                item.status = "raised"
                item.raised_at = segment.start
            if item.status == "raised" and _looks_resolved(segment.text):
                item.status = "resolved"
                item.resolved_at = segment.end
                item.resolution_summary = segment.text.strip()

    def mark_manual_status(self, index: int, status: ChecklistStatus, summary: str | None = None) -> None:
        """Manually set an item status by zero-based index."""
        item = self.items[index]
        item.status = status
        if status == "raised" and item.raised_at is None:
            item.raised_at = time.time()
        if status == "resolved":
            item.resolved_at = time.time()
            item.resolution_summary = summary or item.resolution_summary
        if status == "missed":
            item.suggested_follow_up = summary or build_follow_up_suggestion(item)

    def mark_pending_as_missed(self) -> None:
        """Mark all still-pending items missed at meeting end."""
        for item in self.items:
            if item.status == "pending":
                item.status = "missed"
                item.suggested_follow_up = build_follow_up_suggestion(item)

    def completion_ratio(self) -> float:
        """Return the fraction of items raised or resolved."""
        if not self.items:
            return 1.0
        complete_count = sum(1 for item in self.items if item.status in {"raised", "resolved"})
        return complete_count / len(self.items)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable checklist representation."""
        return {"items": [asdict(item) for item in self.items], "completion_ratio": self.completion_ratio()}

    def render_table(self) -> object:
        """Return a Rich table representing current checklist state."""
        from rich.table import Table

        table = Table(title="Do-Not-Miss Checklist")
        table.add_column("#", justify="right")
        table.add_column("Item")
        table.add_column("Source")
        table.add_column("Status")
        table.add_column("Resolution")
        for index, item in enumerate(self.items, start=1):
            status = _status_badge(item.status)
            resolution = item.resolution_summary or item.suggested_follow_up or ""
            table.add_row(str(index), item.description, item.source, status, resolution)
        return table

    def print_live_panel(self) -> None:
        """Print the current checklist as a terminal panel."""
        try:
            from rich.console import Console
            from rich.panel import Panel

            Console().print(Panel(self.render_table(), border_style="green"))
        except Exception:
            for index, item in enumerate(self.items, start=1):
                print(f"{index}. [{item.status}] {item.description}")


class CommandParser:
    """Handles real-time bang commands typed by the user during a meeting."""

    def __init__(self, session: "MeetingSession") -> None:
        """Keep a reference to the session being controlled."""
        self.session = session

    async def handle(self, raw_command: str) -> str:
        """Parse and execute a command, returning a confirmation message."""
        command = raw_command.strip()
        if not command.startswith("!"):
            return "Ignored note. Commands start with !."
        name, _, argument = command[1:].partition(" ")
        name = name.lower()
        if name == "status":
            return self.session.runtime_memory.get_live_summary()
        if name == "raise":
            return await self._raise_talking_point(argument)
        if name == "tip":
            context = self.session.transcriber.get_recent_context(120) if self.session.transcriber else ""
            tip = self.session.agent.get_advisor_tip(context) if self.session.agent else None
            if tip and self.session.responder:
                self.session.responder.whisper_to_user(tip)
            return tip or "No tip is relevant right now."
        if name in {"last", "transcript"}:
            if not self.session.transcriber:
                return "Transcriber is not ready."
            context = self.session.transcriber.get_recent_context(120)
            return context or "No transcript has been captured yet."
        if name == "listen":
            if not self.session.audio_capture:
                return "Audio capture is not ready."
            stats = self.session.audio_capture.get_stats()
            backend = getattr(self.session.transcriber, "_backend", "unknown") if self.session.transcriber else "unknown"
            segment_count = len(self.session.transcriber.get_full_transcript()) if self.session.transcriber else 0
            return (
                f"capture_started={stats.started}, captured_seconds={stats.captured_seconds:.1f}, "
                f"queued_chunks={stats.queued_chunks}, rms={stats.rms:.1f}, peak={stats.peak}, "
                f"whisper_backend={backend}, transcript_segments={segment_count}"
            )
        if name == "audio":
            if not self.session.responder:
                return "Responder is not ready."
            output_name = self.session.responder.audio_output_device_name or "default speaker"
            monitor = self.session.responder.monitor_default_speaker
            return f"Assistant speech output: {output_name}. Local speaker monitor: {monitor}."
        if name == "store":
            return self.session.store_live_artifacts()
        if name in {"answer", "respond"}:
            if not self.session.agent or not self.session.transcriber:
                return "Agent is not ready."
            context = self.session.transcriber.get_recent_context(120)
            if not context:
                return "No transcript has been captured yet."
            response = self.session.agent.generate_response(context, "manual answer command")
            if self.session.responder and self.session.briefing:
                self.session.responder.speak(response, self.session.briefing.agent_mode)
                self.session.runtime_memory.log_agent_response("manual !answer command", response)
            return response
        if name == "speak":
            if not argument.strip():
                return "Usage: !speak <text>"
            if self.session.responder and self.session.briefing:
                self.session.responder.speak(argument.strip(), self.session.briefing.agent_mode)
                self.session.runtime_memory.log_agent_response("manual !speak command", argument.strip())
                return "Spoken."
            return "Responder is not ready."
        if name == "mute":
            if self.session.agent:
                self.session.agent.mute()
            return "Agent muted."
        if name == "unmute":
            if self.session.agent:
                self.session.agent.unmute()
            return "Agent unmuted."
        if name == "flag":
            if not argument.strip():
                return "Usage: !flag <text>"
            self.session.add_manual_flag(argument.strip())
            return "Flag added."
        if name == "checklist":
            self.session.checklist.print_live_panel()
            return "Checklist printed."
        if name == "whoisspeaking":
            speaker = self.session.speaker_tracker.current_active_speaker() if self.session.speaker_tracker else None
            return f"Current active speaker: {speaker or 'unknown'}"
        if name == "end":
            await self.session.request_stop()
            return "Ending meeting and running post-processing."
        return f"Unknown command: !{name}"

    async def _raise_talking_point(self, argument: str) -> str:
        try:
            point_index = int(argument.strip()) - 1
        except ValueError:
            return "Usage: !raise <n>"
        if point_index < 0 or point_index >= len(self.session.runtime_memory.talking_points_pending):
            return "Talking point number is out of range."
        point = self.session.runtime_memory.talking_points_pending[point_index]
        if self.session.agent:
            response = self.session.agent.generate_response("", f"Talking point is relevant: {point}")
            if self.session.responder and self.session.briefing:
                self.session.responder.speak(response, self.session.briefing.agent_mode)
                self.session.runtime_memory.log_agent_response(f"manual !raise {point_index + 1}", response)
            self.session.agent.memory.mark_talking_point_raised(point)
        if point not in self.session.runtime_memory.talking_points_raised:
            self.session.runtime_memory.talking_points_raised.append(point)
        self.session.runtime_memory.talking_points_pending = [
            item for item in self.session.runtime_memory.talking_points_pending if item != point
        ]
        return f"Raised talking point: {point}"


class MeetingSession:
    """
    Orchestrates one meeting lifecycle: briefing, capture, transcribe, analyze,
    respond, record, stop, pipeline, and artifact summary.
    """

    def __init__(self) -> None:
        """Initialise empty session state."""
        if load_dotenv is not None:
            load_dotenv()
        self.briefing: Briefing | None = None
        self.session_id = ""
        self.audio_capture: AudioCapture | None = None
        self.transcriber: LiveTranscriber | None = None
        self.speaker_tracker: SpeakerTracker | None = None
        self.responder: Responder | None = None
        self.agent: MeetingAgent | None = None
        self.recorder: Recorder | None = None
        self.checklist = DoNotMissChecklist()
        self.runtime_memory = RuntimeMemory()
        self.command_parser = CommandParser(self)
        self._stop_event = asyncio.Event()
        self._tasks: list[asyncio.Task[None]] = []
        self._source: JoinSource = "both"
        self._chunk_offset_seconds = 0.0
        self.focus_mode = os.getenv("ASSISTANT_FOCUS_MODE", "false").strip().lower() in {"1", "true", "yes", "y"}

    async def start(self, briefing: Briefing, source: JoinSource = "both") -> None:
        """Start a live meeting session and run until !end, stop(), or KeyboardInterrupt."""
        self._initialise_components(briefing, source)
        assert self.audio_capture and self.recorder
        self.audio_capture.start(source)
        self.recorder.start_recording(self.session_id)
        self._tasks = [
            asyncio.create_task(self._transcription_loop(), name="transcription_loop"),
            asyncio.create_task(self._command_loop(), name="command_loop"),
            asyncio.create_task(self._comprehension_loop(), name="comprehension_loop"),
        ]
        if not self.focus_mode:
            self._tasks.append(asyncio.create_task(self._checklist_loop(), name="checklist_loop"))
        try:
            await self._stop_event.wait()
        finally:
            await self._cancel_tasks()

    async def start_from_file(self, briefing: Briefing, audio_file: Path | str) -> SessionReport:
        """Process a pre-recorded audio file end to end."""
        self._initialise_components(briefing, "system")
        assert self.recorder and self.transcriber and self.speaker_tracker and self.agent
        self.recorder.start_recording(self.session_id)
        wav_path = normalize_audio_file_to_wav(Path(audio_file))
        for chunk_bytes, offset in _iter_wav_chunks(wav_path, int(os.getenv("AUDIO_CHUNK_SECONDS", "30"))):
            self.recorder.write_chunk(chunk_bytes)
            speaker_map = self.speaker_tracker.process_chunk(chunk_bytes, offset)
            segments = self.transcriber.transcribe_chunk(chunk_bytes, offset)
            assignments = assign_speakers_to_segments(speaker_map, [segment.start for segment in segments])
            for segment in segments:
                segment.speaker_label = assignments.get(segment.start, segment.speaker_label)
                self.checklist.update_from_segment(segment)
                self.agent.on_new_transcript_segment(segment)
        return await self.stop()

    async def stop(self) -> SessionReport:
        """Stop live capture, save artifacts, run the pipeline, and return a report summary."""
        await self.request_stop()
        await self._cancel_tasks()
        if self.audio_capture:
            self.audio_capture.stop()
        if self.recorder is None or self.briefing is None:
            raise RuntimeError("Session was not started.")
        audio_mp3_path = self.recorder.stop_recording()
        self.checklist.mark_pending_as_missed()
        transcript = self.transcriber.get_full_transcript() if self.transcriber else self.runtime_memory.full_transcript
        if not self.runtime_memory.full_transcript:
            self.runtime_memory.full_transcript = list(transcript)
        transcript_paths = self.recorder.save_transcript(transcript, self.session_id)
        metadata_path = self.recorder.save_session_metadata(self.briefing, self.runtime_memory, self.session_id, self.checklist)
        from post_meeting.pipeline_runner import run_existing_pipeline

        pipeline_outputs = run_existing_pipeline(self.session_id)
        report = SessionReport(
            session_id=self.session_id,
            session_dir=self.recorder.recordings_dir / self.session_id,
            audio_mp3_path=audio_mp3_path,
            transcript_paths=transcript_paths,
            metadata_path=metadata_path,
            pipeline_outputs=pipeline_outputs,
            duration_seconds=_duration_seconds(transcript),
            speaker_count=len({segment.speaker_label for segment in transcript if segment.speaker_label}),
            word_count=sum(len(segment.text.split()) for segment in transcript),
            live_summary=self.runtime_memory.get_live_summary(),
        )
        await self._try_feature8_report_hooks(report)
        return report

    async def request_stop(self) -> None:
        """Ask all loops to stop gracefully."""
        self._stop_event.set()

    def add_manual_flag(self, text: str) -> None:
        """Add a user-supplied key moment to runtime memory."""
        from assistant.memory import KeyMoment

        timestamp = self.runtime_memory.full_transcript[-1].end if self.runtime_memory.full_transcript else time.time()
        self.runtime_memory.key_moments.append(KeyMoment(timestamp, text, "user_flag", "medium"))

    def store_live_artifacts(self) -> str:
        """Persist the current transcript and metadata without ending the session."""
        if self.recorder is None or self.briefing is None:
            return "Session was not started."
        transcript = self.transcriber.get_full_transcript() if self.transcriber else self.runtime_memory.full_transcript
        if not transcript:
            return "No transcript has been captured yet."
        transcript_paths = self.recorder.save_transcript(transcript, self.session_id)
        metadata_path = self.recorder.save_session_metadata(self.briefing, self.runtime_memory, self.session_id, self.checklist)
        return (
            f"Stored {len(transcript)} transcript segments. "
            f"Transcript: {transcript_paths.get('txt')}. Metadata: {metadata_path}"
        )

    def _initialise_components(self, briefing: Briefing, source: JoinSource) -> None:
        self.briefing = briefing
        self.session_id = generate_session_id(briefing.meeting_title)
        self._source = source
        self._stop_event = asyncio.Event()
        self.audio_capture = AudioCapture()
        self.transcriber = LiveTranscriber()
        self.speaker_tracker = SpeakerTracker()
        self.responder = Responder(self.audio_capture)
        self.recorder = Recorder()
        self.checklist = DoNotMissChecklist.from_briefing(briefing)
        self.runtime_memory = RuntimeMemory(talking_points_pending=list(briefing.talking_points))
        self.focus_mode = os.getenv("ASSISTANT_FOCUS_MODE", "false").strip().lower() in {"1", "true", "yes", "y"}
        memory = learn_from_briefing(briefing)
        self.agent = MeetingAgent(
            briefing,
            memory,
            self.transcriber,
            self.speaker_tracker,
            self.responder,
            self.runtime_memory,
        )
        if self.focus_mode:
            self.agent.mute()
        self.command_parser = CommandParser(self)

    async def _transcription_loop(self) -> None:
        assert self.audio_capture and self.transcriber and self.speaker_tracker and self.agent and self.recorder
        while not self._stop_event.is_set():
            chunk = self.audio_capture.get_chunk()
            if chunk is None:
                await asyncio.sleep(0.25)
                continue
            self.recorder.write_chunk(chunk)
            offset = self._chunk_offset_seconds
            speaker_map = self.speaker_tracker.process_chunk(chunk, offset)
            segments = self.transcriber.transcribe_chunk(chunk, offset)
            assignments = assign_speakers_to_segments(speaker_map, [segment.start for segment in segments])
            for segment in segments:
                segment.speaker_label = assignments.get(segment.start, segment.speaker_label)
                self.checklist.update_from_segment(segment)
                self.agent.on_new_transcript_segment(segment)
            self._chunk_offset_seconds += max(1, self.audio_capture.chunk_seconds - self.audio_capture.overlap_seconds)

    async def _command_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._stop_event.is_set():
            try:
                line = await loop.run_in_executor(None, input, "")
            except (EOFError, KeyboardInterrupt):
                await self.request_stop()
                return
            if not line:
                continue
            message = await self.command_parser.handle(line)
            if message:
                print(message)

    async def _checklist_loop(self) -> None:
        while not self._stop_event.is_set():
            self.checklist.print_live_panel()
            await asyncio.sleep(10)

    async def _comprehension_loop(self) -> None:
        while not self._stop_event.is_set():
            await asyncio.sleep(max(1, self.runtime_memory.memory_update_interval_seconds))
            if self.runtime_memory.full_transcript:
                self.runtime_memory.get_live_summary()

    async def _cancel_tasks(self) -> None:
        current = asyncio.current_task()
        for task in self._tasks:
            if task is current or task.done():
                continue
            task.cancel()
        for task in self._tasks:
            if task is current:
                continue
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks = []

    async def _try_feature8_report_hooks(self, report: SessionReport) -> None:
        with contextlib.suppress(Exception):
            from post_meeting.report_builder import build_report
            from post_meeting.notifier import send_report

            built_report = build_report(self.session_id, self.runtime_memory, self.briefing, report.pipeline_outputs)
            send_report(built_report)


def build_follow_up_suggestion(item: ChecklistItem) -> str:
    """Create a short follow-up suggestion for a missed checklist item."""
    if item.source == "talking_point":
        return f"Send a follow-up note raising this point explicitly: {item.description}"
    if item.source == "objective":
        return f"Follow up with the group to confirm the objective was resolved: {item.description}"
    return f"Follow up on this instruction: {item.description}"


def _iter_wav_chunks(wav_path: Path, chunk_seconds: int) -> list[tuple[bytes, float]]:
    chunks: list[tuple[bytes, float]] = []
    with wave.open(str(wav_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frames_per_chunk = max(1, sample_rate * chunk_seconds)
        offset = 0.0
        while True:
            frames = wav_file.readframes(frames_per_chunk)
            if not frames:
                break
            chunks.append((_frames_to_wav(frames, sample_rate, channels, sample_width), offset))
            offset += len(frames) / max(1, sample_rate * channels * sample_width)
    return chunks


def _frames_to_wav(frames: bytes, sample_rate: int, channels: int, sample_width: int) -> bytes:
    import io

    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)
    return output.getvalue()


def _duration_seconds(segments: list[TranscriptSegment]) -> float:
    if not segments:
        return 0.0
    return max(segment.end for segment in segments) - min(segment.start for segment in segments)


def _semanticish_match(left: str, right: str) -> float:
    left_tokens = set(_tokens(left))
    right_tokens = set(_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    token_score = len(left_tokens & right_tokens) / len(left_tokens)
    sequence_score = SequenceMatcher(None, left.lower(), right.lower()).ratio()
    return max(token_score, sequence_score)


def _tokens(text: str) -> list[str]:
    import re

    stopwords = {"about", "should", "would", "could", "there", "their", "meeting", "discuss"}
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in stopwords]


def _looks_resolved(text: str) -> bool:
    lowered = text.lower()
    markers = ["agreed", "decided", "resolved", "confirmed", "we'll", "we will", "the plan is"]
    return any(marker in lowered for marker in markers)


def _looks_like_checklist_instruction(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in ["make sure", "do not miss", "must cover", "ensure", "confirm"])


def _status_badge(status: ChecklistStatus) -> str:
    colors = {
        "pending": "[yellow]pending[/yellow]",
        "raised": "[cyan]raised[/cyan]",
        "resolved": "[green]resolved[/green]",
        "missed": "[red]missed[/red]",
    }
    return colors[status]
