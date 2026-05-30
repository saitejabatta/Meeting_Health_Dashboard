# assistant/agent.py
"""Core AI meeting agent decision logic and response generation."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Literal

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from assistant.briefing import AgentMode, Briefing
from assistant.memory import (
    AgentMemory,
    KeyMoment,
    RuntimeMemory,
    cosine_similarity,
    detect_action_item,
    detect_decision_text,
    estimate_valence,
)
from assistant.responder import Responder
from assistant.speaker_tracker import SpeakerTracker
from assistant.transcriber import LiveTranscriber, TranscriptSegment

logger = logging.getLogger(__name__)


@dataclass
class SpeakDecision:
    """Decision about whether the agent should respond."""

    should_speak: bool
    reason: str
    urgency: Literal["high", "medium", "low", "none"]


@dataclass
class Flag:
    """A real-time alert surfaced privately to the user."""

    type: str
    message: str
    timestamp: float
    severity: Literal["low", "medium", "high"]


class MeetingAgent:
    """
    Central AI agent for live meeting participation.

    It keeps briefing context, consumes transcript segments, updates runtime
    memory, decides whether to speak, and generates responses with rule-based
    fallback when LLM calls fail.
    """

    def __init__(
        self,
        briefing: Briefing,
        memory: AgentMemory,
        transcriber: LiveTranscriber,
        speaker_tracker: SpeakerTracker,
        responder: Responder,
        runtime_memory: RuntimeMemory | None = None,
    ) -> None:
        """Initialise the agent brain and runtime counters."""
        if load_dotenv is not None:
            load_dotenv()
        self.briefing = briefing
        self.memory = memory
        self.transcriber = transcriber
        self.speaker_tracker = speaker_tracker
        self.responder = responder
        self.runtime_memory = runtime_memory or RuntimeMemory(talking_points_pending=list(briefing.talking_points))
        self.speaking_turns_used = 0
        self.last_segment_at = 0.0
        self.muted = False
        self._flag_history: list[Flag] = []
        self._last_topic: str | None = None

    def on_new_transcript_segment(self, segment: TranscriptSegment) -> None:
        """Handle a new transcript segment and respond if the briefing allows it."""
        self.runtime_memory.update(segment)
        self.last_segment_at = max(self.last_segment_at, segment.end)
        context = self.transcriber.get_recent_context(120)
        for flag in self.analyze_for_flags(segment):
            self.runtime_memory.key_moments.append(KeyMoment(flag.timestamp, flag.message, flag.type, flag.severity))
            self.responder.whisper_to_user(flag.message)
        if self.briefing.agent_mode == AgentMode.ADVISOR:
            tip = self.get_advisor_tip(context)
            if tip:
                self.responder.whisper_to_user(tip)
        if self.muted or not self.briefing.auto_respond:
            return
        decision = self.should_speak(context)
        if not decision.should_speak:
            return
        response = self.generate_response(context, decision.reason)
        if response:
            self.responder.speak(response, self.briefing.agent_mode)
            self.speaking_turns_used += 1
            self.runtime_memory.log_agent_response(decision.reason, response)
            self._mark_relevant_talking_point_raised(context)

    def should_speak(self, context: str) -> SpeakDecision:
        """Return a speaking decision based on mode, limits, avoid topics, and conversation context."""
        if self.briefing.agent_mode == AgentMode.SILENT_OBSERVER:
            return SpeakDecision(False, "Silent observer mode", "none")
        if self.muted:
            return SpeakDecision(False, "Agent is muted", "none")
        if self.briefing.max_speaking_turns and self.speaking_turns_used >= self.briefing.max_speaking_turns:
            return SpeakDecision(False, "Maximum speaking turns reached", "none")
        avoided, topic, score = self.memory.is_topic_avoided(context)
        if avoided:
            return SpeakDecision(False, f"Avoided topic detected: {topic} ({score:.2f})", "none")
        talking_point = self._best_relevant_talking_point(context)
        if talking_point is not None:
            return SpeakDecision(True, f"Talking point is relevant: {talking_point}", "high")
        if _looks_like_direct_question(context) and self.briefing.agent_mode in {AgentMode.FULL_PROXY, AgentMode.PARTICIPATOR}:
            return SpeakDecision(True, "A direct question was asked", "high")
        if self._conversation_silent_for(8) and self.memory.talking_points_queue:
            return SpeakDecision(True, f"Silence gap; pending talking point: {self.memory.talking_points_queue[0]}", "medium")
        return SpeakDecision(False, "No response needed", "none")

    def generate_response(self, context: str, trigger: str) -> str:
        """Generate a natural first-person response for the current context and trigger."""
        avoided, topic, _ = self.memory.is_topic_avoided(context)
        if avoided:
            return f"I'd rather keep us focused away from {topic} and stay with the main objective."
        llm_response = self._generate_response_with_llm(context, trigger)
        if llm_response:
            return _sanitize_agent_response(llm_response)
        return self._generate_response_with_rules(context, trigger)

    def get_advisor_tip(self, context: str) -> str | None:
        """Return a short private suggestion for Advisor mode, if useful."""
        avoided, topic, _ = self.memory.is_topic_avoided(context)
        if avoided:
            return f"Conversation is near an avoided topic: {topic}. Deflect or stay quiet."
        talking_point = self._best_relevant_talking_point(context)
        if talking_point:
            return f"Now is a good time to raise: {talking_point}"
        if _looks_like_direct_question(context):
            return "A direct question is on the table. Consider answering briefly, then checking for alignment."
        return None

    def analyze_for_flags(self, segment: TranscriptSegment) -> list[Flag]:
        """Analyze a new segment for decisions, actions, open questions, tension, topic shifts, and opportunities."""
        flags: list[Flag] = []
        text = segment.text.strip()
        if not text:
            return flags

        decision = detect_decision_text(text)
        if decision and self._confirm_with_llm_or_rules("decision", text):
            flags.append(Flag("decision", f"Decision detected: {decision}", segment.start, "high"))

        action_item = detect_action_item(text)
        if action_item is not None and self._confirm_with_llm_or_rules("action_item", text):
            assignee = action_item.who if action_item.who != "unassigned" else "Unassigned"
            due = f" by {action_item.by_when}" if action_item.by_when else ""
            flags.append(Flag("action_item", f"Action item: {assignee} to {action_item.what}{due}", segment.start, "high"))

        open_question = self._find_open_question(segment)
        if open_question:
            flags.append(Flag("open_question", f"Open question - may need follow-up: {open_question}", segment.start, "medium"))

        if self._has_tension_spike():
            flags.append(Flag("tension_spike", "Tension detected. Consider a re-framing statement.", segment.start, "high"))

        new_topic = self._detect_topic_shift(text)
        if new_topic:
            flags.append(Flag("topic_shift", f"Topic shifted to: {new_topic}", segment.start, "low"))

        talking_point = self._best_relevant_talking_point(self.transcriber.get_recent_context(120))
        if talking_point:
            flags.append(Flag("talking_point_opportunity", f"Good moment to raise: {talking_point}", segment.start, "medium"))

        return self._dedupe_flags(flags)

    def mute(self) -> None:
        """Disable autonomous speaking."""
        self.muted = True

    def unmute(self) -> None:
        """Enable autonomous speaking according to the briefing."""
        self.muted = False

    def _generate_response_with_llm(self, context: str, trigger: str) -> str | None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        try:
            from openai import OpenAI
        except ImportError:
            return None
        prompt = {
            "context": context,
            "trigger": trigger,
            "instruction": (
                "Respond as the user in first person. Match the response style. "
                "Do not mention being an AI or assistant. Keep it suitable to say aloud in a meeting."
            ),
        }
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_AGENT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
                messages=[
                    {"role": "system", "content": self.memory.system_prompt},
                    {"role": "user", "content": json.dumps(prompt)},
                ],
                temperature=0.3,
                timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "20")),
            )
            return response.choices[0].message.content or None
        except Exception as exc:
            logger.warning("Agent response LLM failed; using rule-based fallback: %s", exc)
            return None

    def _generate_response_with_rules(self, context: str, trigger: str) -> str:
        talking_point = self._extract_talking_point_from_trigger(trigger) or self._best_relevant_talking_point(context)
        if talking_point:
            return f"I want to make sure we cover {talking_point} before we move on."
        if "direct question" in trigger.lower():
            return "My view is that we should stay aligned to the objective and make the next step explicit."
        if self.memory.relevant_background(context):
            return "One bit of context that may help: this connects back to the background we discussed before the meeting."
        return "I think the useful next step is to clarify the decision and owner before we move on."

    def _best_relevant_talking_point(self, context: str) -> str | None:
        if not context.strip():
            return self.memory.talking_points_queue[0] if self._conversation_silent_for(8) and self.memory.talking_points_queue else None
        context_embedding = self.memory.embedding_provider.embed(context)
        best_point: str | None = None
        best_score = 0.0
        for point in self.memory.talking_points_queue:
            lexical_score = _token_overlap_score(context, point)
            semantic_score = cosine_similarity(context_embedding, self.memory.embedding_provider.embed(point))
            score = max(lexical_score, semantic_score)
            if score > best_score:
                best_point = point
                best_score = score
        return best_point if best_score > 0.7 else None

    def _mark_relevant_talking_point_raised(self, context: str) -> None:
        point = self._best_relevant_talking_point(context)
        if point is None:
            point = self._extract_talking_point_from_trigger(context)
        if point:
            self.memory.mark_talking_point_raised(point)
            if point not in self.runtime_memory.talking_points_raised:
                self.runtime_memory.talking_points_raised.append(point)
            self.runtime_memory.talking_points_pending = list(self.memory.talking_points_queue)

    def _extract_talking_point_from_trigger(self, trigger: str) -> str | None:
        prefix = "Talking point is relevant:"
        if trigger.startswith(prefix):
            return trigger[len(prefix) :].strip()
        prefix = "Silence gap; pending talking point:"
        if trigger.startswith(prefix):
            return trigger[len(prefix) :].strip()
        return None

    def _conversation_silent_for(self, seconds: float) -> bool:
        if self.last_segment_at <= 0:
            return False
        return time.time() - self.last_segment_at >= seconds

    def _confirm_with_llm_or_rules(self, flag_type: str, text: str) -> bool:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return True
        try:
            from openai import OpenAI
        except ImportError:
            return True
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_FLAG_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
                messages=[
                    {"role": "system", "content": "Return JSON only: {\"confirmed\": true|false}."},
                    {"role": "user", "content": f"Is this transcript segment a real {flag_type}? Segment: {text}"},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "20")),
            )
            payload = json.loads(response.choices[0].message.content or "{}")
            return bool(payload.get("confirmed", True))
        except Exception as exc:
            logger.warning("Flag confirmation failed; accepting rule-based %s flag: %s", flag_type, exc)
            return True

    def _find_open_question(self, segment: TranscriptSegment) -> str | None:
        if "?" in segment.text:
            return None
        now = segment.end
        recent_questions = [
            question for question in self.runtime_memory.questions_raised
            if question not in self.runtime_memory.questions_answered
        ]
        if not recent_questions:
            return None
        question_segments = [
            item for item in self.runtime_memory.full_transcript
            if "?" in item.text and item.text.strip() in recent_questions and now - item.end >= 60
        ]
        if not question_segments:
            return None
        question = question_segments[0].text.strip()
        self.runtime_memory.questions_answered.append(question)
        return question

    def _has_tension_spike(self) -> bool:
        recent = self.runtime_memory.full_transcript[-3:]
        if len(recent) < 3:
            return False
        return sum(estimate_valence(segment.text) for segment in recent) / 3 < -0.4

    def _detect_topic_shift(self, text: str) -> str | None:
        topic = _local_topic_label(text)
        if not topic:
            return None
        if self._last_topic is None:
            self._last_topic = topic
            return None
        if _token_overlap_score(topic, self._last_topic) < 0.35:
            self._last_topic = topic
            return topic
        return None

    def _dedupe_flags(self, flags: list[Flag]) -> list[Flag]:
        unique: list[Flag] = []
        for flag in flags:
            duplicate = any(
                existing.type == flag.type
                and abs(existing.timestamp - flag.timestamp) < 90
                and _token_overlap_score(existing.message, flag.message) > 0.7
                for existing in self._flag_history
            )
            if not duplicate:
                self._flag_history.append(flag)
                unique.append(flag)
        return unique


def _looks_like_direct_question(context: str) -> bool:
    lines = [line.strip() for line in context.splitlines() if line.strip()]
    if not lines:
        return False
    recent = " ".join(lines[-3:]).lower()
    if "?" in recent:
        return True
    return bool(re.search(r"\b(what do you think|any thoughts|can you|could you|do you agree|your view)\b", recent))


def _token_overlap_score(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9]+", left.lower()))
    right_tokens = set(re.findall(r"[a-z0-9]+", right.lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(right_tokens)


def _sanitize_agent_response(text: str) -> str:
    cleaned = re.sub(r"\b(as an ai|i am an ai assistant|i'm an ai assistant)\b[:,]?\s*", "", text, flags=re.IGNORECASE)
    return cleaned.strip().strip('"')


def _local_topic_label(text: str) -> str:
    stopwords = {
        "about",
        "after",
        "again",
        "could",
        "going",
        "meeting",
        "should",
        "their",
        "there",
        "think",
        "would",
    }
    tokens = [token for token in re.findall(r"[a-z][a-z0-9-]+", text.lower()) if len(token) > 4 and token not in stopwords]
    return " ".join(tokens[:3])
