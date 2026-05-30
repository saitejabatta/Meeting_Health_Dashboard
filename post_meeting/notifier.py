# post_meeting/notifier.py
"""Notification delivery for completed meeting reports."""

from __future__ import annotations

import os
from html import escape

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from post_meeting.report_builder import MeetingReport


def send_report(report: MeetingReport) -> None:
    """Send a report through configured channels, or print a terminal summary."""
    if load_dotenv is not None:
        load_dotenv()
    sent = False
    if os.getenv("SLACK_WEBHOOK_URL"):
        sent = _send_slack(report) or sent
    if os.getenv("SENDGRID_API_KEY") and os.getenv("REPORT_EMAIL"):
        sent = _send_email(report) or sent
    if not sent:
        _print_terminal_summary(report)


def _send_slack(report: MeetingReport) -> bool:
    try:
        from slack_sdk.webhook import WebhookClient

        payload = report.payload
        decisions = payload.get("key_decisions", [])[:3]
        action_count = len(payload.get("action_items", []))
        health_score = (payload.get("metrics") or {}).get("meeting_health_score", "N/A")
        text = (
            f"Meeting report ready: {report.session_id}\n"
            f"Health score: {health_score}\n"
            f"Action items: {action_count}\n"
            f"Top decisions: {', '.join(decisions) if decisions else 'none'}\n"
            f"Report: {report.markdown_path}"
        )
        response = WebhookClient(os.environ["SLACK_WEBHOOK_URL"]).send(text=text)
        return response.status_code < 400
    except Exception as exc:
        print(f"Slack notification failed: {exc}")
        return False


def _send_email(report: MeetingReport) -> bool:
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Content, Email, Mail, To

        sender = os.getenv("REPORT_EMAIL_FROM", os.getenv("REPORT_EMAIL", "reports@example.com"))
        recipient = os.environ["REPORT_EMAIL"]
        markdown = report.markdown_path.read_text()
        html = "<pre style='white-space: pre-wrap; font-family: system-ui, sans-serif'>" + escape(markdown) + "</pre>"
        message = Mail(
            from_email=Email(sender),
            to_emails=To(recipient),
            subject=f"Meeting report: {report.session_id}",
            plain_text_content=markdown,
            html_content=Content("text/html", html),
        )
        response = SendGridAPIClient(os.environ["SENDGRID_API_KEY"]).send(message)
        return response.status_code < 400
    except Exception as exc:
        print(f"Email notification failed: {exc}")
        return False


def _print_terminal_summary(report: MeetingReport) -> None:
    print("\nMeeting report ready")
    print(f"Session: {report.session_id}")
    print(f"Summary: {report.summary}")
    print(f"Markdown: {report.markdown_path}")
    print(f"JSON: {report.json_path}\n")
