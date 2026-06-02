# assistant/cli.py
"""Command-line entry point for the AI Meeting Assistant."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from assistant.briefing import Briefing, collect_briefing_interactive, parse_briefing_from_text
from assistant.session import MeetingSession, SessionReport


def main() -> None:
    """Run the interactive command-line meeting assistant."""
    args = _parse_args()
    try:
        asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        print("\nInterrupted. Session cleanup is handled inside the active meeting loop when running.")


async def _async_main(args: argparse.Namespace) -> None:
    console = _console()
    console.print("[bold cyan]AI Meeting Assistant[/bold cyan]")
    briefing = _load_briefing(args.briefing_file)
    while briefing is None:
        raw = input("Brief me on your meeting, type 'paste' for multi-line, 'file:path', or 'wizard'\n> ").strip()
        briefing = _briefing_from_initial_input(raw)
    _print_briefing_summary(briefing)
    confirmation = input("Does this look correct? (yes / edit / wizard)\n> ").strip().lower()
    if confirmation in {"wizard", "w"}:
        briefing = collect_briefing_interactive()
    elif confirmation not in {"yes", "y"}:
        replacement = _collect_multiline(
            "Paste the corrected full briefing. End with a line containing only END.\n"
        )
        briefing = parse_briefing_from_text(replacement)
        _print_briefing_summary(briefing)
        confirmation = input("Use this briefing? (yes / wizard)\n> ").strip().lower()
        if confirmation not in {"yes", "y"}:
            briefing = collect_briefing_interactive()

    join_choice = input(
        "How are you joining the meeting? [local mic / zoom / teams / google meet / audio file]\n> "
    ).strip().lower()
    session = MeetingSession()
    if join_choice in {"audio file", "file", "recording"}:
        path = Path(input("Audio file path\n> ").strip()).expanduser()
        report = await session.start_from_file(briefing, path)
        _print_session_report(report)
        return

    source = "both"
    if join_choice in {"zoom", "teams", "google meet", "gmeet"}:
        meeting_url = input("Meeting URL\n> ").strip()
        _launch_virtual_meeting(
            join_choice,
            meeting_url,
            briefing.assistant_display_name or _transparent_display_name(briefing.user_name),
        )
        input("After the meeting app opens and you are joined with audio connected, press Enter to start capture.\n> ")
        source = "system"
    elif join_choice in {"local mic", "mic", "local"}:
        source = "mic"

    try:
        await session.start(briefing, source=source)
    except Exception as exc:
        _console().print(f"[red]Could not start meeting session: {exc}[/red]")
        return
    finally:
        if session.recorder is not None and session.recorder.audio_wav_path is not None:
            report = await session.stop()
            _print_session_report(report)


def _print_briefing_summary(briefing: Briefing) -> None:
    console = _console()
    console.print("[bold]Briefing Summary[/bold]")
    for key, value in briefing.__dict__.items():
        console.print(f"{key}: {value}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AI Meeting Assistant.")
    parser.add_argument("--briefing-file", type=Path, help="Path to a saved briefing text file.")
    return parser.parse_args()


def _load_briefing(path: Path | None) -> Briefing | None:
    if path is None:
        return None
    return parse_briefing_from_text(path.expanduser().read_text(encoding="utf-8"))


def _briefing_from_initial_input(raw: str) -> Briefing | None:
    if raw.lower() == "wizard":
        return collect_briefing_interactive()
    if raw.lower() in {"paste", "multiline"}:
        return parse_briefing_from_text(
            _collect_multiline("Paste the full briefing. End with a line containing only END.\n")
        )
    if raw.startswith("file:"):
        path = Path(raw.partition(":")[2].strip()).expanduser()
        return parse_briefing_from_text(path.read_text(encoding="utf-8"))
    if raw.startswith("@"):
        path = Path(raw[1:].strip()).expanduser()
        return parse_briefing_from_text(path.read_text(encoding="utf-8"))
    if raw:
        return parse_briefing_from_text(raw)
    return None


def _collect_multiline(prompt: str) -> str:
    print(prompt, end="")
    lines: list[str] = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def _print_virtual_meeting_fallback(choice: str) -> None:
    console = _console()
    console.print(
        f"[yellow]{choice.title()} automation is not configured yet.[/yellow] "
        "Join manually and route meeting audio to your configured virtual audio device. "
        "The assistant will capture system audio."
    )


def _launch_virtual_meeting(choice: str, meeting_url: str, display_name: str = "") -> None:
    console = _console()
    try:
        if choice == "zoom":
            from integrations.zoom_bot import ZoomBot

            result = ZoomBot().join(meeting_url, display_name=display_name)
        elif choice in {"google meet", "gmeet"}:
            from integrations.gmeet_bot import GoogleMeetBot

            result = GoogleMeetBot().join(meeting_url, display_name=display_name)
        else:
            from integrations.teams_bot import TeamsBot

            result = TeamsBot().join(meeting_url, display_name=display_name)
        style = "green" if result.joined else "yellow"
        console.print(f"[{style}]{result.message}[/{style}]")
    except Exception as exc:
        console.print(f"[yellow]Could not launch {choice}: {exc}. Join manually and route audio to the virtual device.[/yellow]")


def _transparent_display_name(user_name: str) -> str:
    clean_name = user_name.strip() or "User"
    words = {piece for piece in clean_name.lower().replace("-", " ").split() if piece}
    if "assistant" in words or "ai" in words:
        return clean_name
    return f"{clean_name} - AI Assistant"


def _print_session_report(report: SessionReport) -> None:
    console = _console()
    console.print("[bold green]Session complete[/bold green]")
    console.print(f"Session: {report.session_id}")
    console.print(f"Directory: {report.session_dir}")
    console.print(f"Audio: {report.audio_mp3_path}")
    console.print(f"Metadata: {report.metadata_path}")
    console.print(f"Pipeline: {report.pipeline_outputs}")
    console.print(f"Summary: {report.live_summary}")


def _console() -> object:
    try:
        from rich.console import Console

        return Console()
    except Exception:
        return _PlainConsole()


class _PlainConsole:
    """Tiny print-compatible fallback when Rich is unavailable."""

    def print(self, message: object) -> None:
        """Print a message without styling."""
        print(str(message))


if __name__ == "__main__":
    main()
