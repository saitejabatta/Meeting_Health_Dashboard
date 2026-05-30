# assistant/cli.py
"""Command-line entry point for the AI Meeting Assistant."""

from __future__ import annotations

import asyncio
from pathlib import Path

from assistant.briefing import Briefing, collect_briefing_interactive, parse_briefing_from_text
from assistant.session import MeetingSession, SessionReport


def main() -> None:
    """Run the interactive command-line meeting assistant."""
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        print("\nInterrupted. Session cleanup is handled inside the active meeting loop when running.")


async def _async_main() -> None:
    console = _console()
    console.print("[bold cyan]AI Meeting Assistant[/bold cyan]")
    raw = input("Brief me on your meeting, or type 'wizard' to go step by step\n> ").strip()
    if raw.lower() == "wizard":
        briefing = collect_briefing_interactive()
    else:
        briefing = parse_briefing_from_text(raw)
        _print_briefing_summary(briefing)
        confirmation = input("Does this look correct? (yes / edit)\n> ").strip().lower()
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
        _launch_virtual_meeting(join_choice, meeting_url)
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


def _print_virtual_meeting_fallback(choice: str) -> None:
    console = _console()
    console.print(
        f"[yellow]{choice.title()} automation is not configured yet.[/yellow] "
        "Join manually and route meeting audio to your configured virtual audio device. "
        "The assistant will capture system audio."
    )


def _launch_virtual_meeting(choice: str, meeting_url: str) -> None:
    console = _console()
    try:
        if choice == "zoom":
            from integrations.zoom_bot import ZoomBot

            result = ZoomBot().join(meeting_url)
        elif choice in {"google meet", "gmeet"}:
            from integrations.gmeet_bot import GoogleMeetBot

            result = GoogleMeetBot().join(meeting_url)
        else:
            from integrations.teams_bot import TeamsBot

            result = TeamsBot().join(meeting_url)
        style = "green" if result.joined else "yellow"
        console.print(f"[{style}]{result.message}[/{style}]")
    except Exception as exc:
        console.print(f"[yellow]Could not launch {choice}: {exc}. Join manually and route audio to the virtual device.[/yellow]")


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
