# assistant/memory.py
"""Briefing-derived memory used by the AI Meeting Assistant."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Protocol

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from assistant.briefing import AgentMode, Briefing
from assistant.transcriber import TranscriptSegment

logger = logging.getLogger(__name__)


class VectorStore(Protocol):
    """Protocol for in-memory stores that can return relevant text chunks."""

    def add_texts(self, texts: list[str]) -> None:
        """Add texts to the vector store."""

    def similarity_search(self, query: str, k: int = 3) -> list[str]:
        """Return the most similar stored texts for a query."""


@dataclass
class EmbeddingRecord:
    """A text value and the embedding vector used for semantic comparison."""

    text: str
    embedding: list[float]


@dataclass
class KeyMoment:
    """An important moment detected during the meeting."""

    timestamp: float
    text: str
    type: str
    importance: str


@dataclass
class ActionItem:
    """An action item detected from the live transcript."""

    what: str
    who: str
    by_when: str | None
    confidence: float
    source: str = "agent-detected"


@dataclass
class AgentResponseLog:
    """A response the agent gave during the meeting."""

    timestamp: float
    trigger: str
    text: str


@dataclass
class ComprehensionResult:
    """Structured understanding extracted from a transcript window."""

    key_points: list[str]
    decisions: list[str]
    action_items: list[ActionItem]
    open_questions: list[str]
    sentiment_summary: str
    notable_quotes: list[str]
    topics: list[str]


@dataclass
class AgentMemory:
    """Structured memory produced from the pre-meeting briefing."""

    system_prompt: str
    talking_points_queue: list[str]
    raised_points_log: list[str]
    topics_to_avoid_embeddings: list[EmbeddingRecord]
    background_vector_store: VectorStore
    briefing: Briefing
    embedding_provider: "EmbeddingProvider"

    def is_topic_avoided(self, text: str, threshold: float = 0.75) -> tuple[bool, str | None, float]:
        """Check whether text is semantically similar to any avoided topic."""
        if not text.strip() or not self.topics_to_avoid_embeddings:
            return False, None, 0.0
        lexical_match = _find_lexical_topic_match(text, [record.text for record in self.topics_to_avoid_embeddings])
        if lexical_match is not None:
            return True, lexical_match, 1.0
        query_embedding = self.embedding_provider.embed(text)
        best_topic: str | None = None
        best_score = 0.0
        for record in self.topics_to_avoid_embeddings:
            score = cosine_similarity(query_embedding, record.embedding)
            if score > best_score:
                best_topic = record.text
                best_score = score
        return best_score > threshold, best_topic, best_score

    def mark_talking_point_raised(self, point: str) -> None:
        """Move a talking point from pending queue to raised log."""
        normalized = point.strip()
        if not normalized:
            return
        self.talking_points_queue = [item for item in self.talking_points_queue if item != normalized]
        if normalized not in self.raised_points_log:
            self.raised_points_log.append(normalized)

    def relevant_background(self, query: str, k: int = 3) -> list[str]:
        """Return briefing background chunks relevant to the current conversation."""
        return self.background_vector_store.similarity_search(query, k=k)


@dataclass
class RuntimeMemory:
    """Tracks everything that happens during the meeting."""

    full_transcript: list[TranscriptSegment] = field(default_factory=list)
    key_moments: list[KeyMoment] = field(default_factory=list)
    decisions_detected: list[str] = field(default_factory=list)
    action_items_detected: list[ActionItem] = field(default_factory=list)
    questions_raised: list[str] = field(default_factory=list)
    questions_answered: list[str] = field(default_factory=list)
    talking_points_raised: list[str] = field(default_factory=list)
    talking_points_pending: list[str] = field(default_factory=list)
    topics_discussed: list[str] = field(default_factory=list)
    sentiment_timeline: list[dict[str, float]] = field(default_factory=list)
    agent_responses_given: list[AgentResponseLog] = field(default_factory=list)
    memory_update_interval_seconds: int = field(default_factory=lambda: int(os.getenv("MEMORY_UPDATE_INTERVAL_SECONDS", "300")))
    last_comprehension_at: float = 0.0
    cached_live_summary: str = ""
    cached_live_summary_at: float = 0.0

    def update(self, segment: TranscriptSegment) -> None:
        """Update runtime memory with a new transcript segment."""
        if not segment.text.strip():
            return
        self.full_transcript.append(segment)
        self.full_transcript.sort(key=lambda item: (item.start, item.end))
        self.sentiment_timeline.append({"timestamp": segment.start, "valence": estimate_valence(segment.text)})
        if "?" in segment.text:
            self._append_unique(self.questions_raised, segment.text.strip())
        decision = detect_decision_text(segment.text)
        if decision:
            self._append_unique(self.decisions_detected, decision)
            self._append_key_moment(KeyMoment(segment.start, decision, "decision", "high"))
        action_item = detect_action_item(segment.text)
        if action_item is not None:
            self._append_action_item(action_item)
            self._append_key_moment(KeyMoment(segment.start, action_item.what, "action_item", "medium"))

        now = time.time()
        if now - self.last_comprehension_at >= self.memory_update_interval_seconds:
            window = recent_segments(self.full_transcript, seconds=self.memory_update_interval_seconds)
            result = run_comprehension_pass(window)
            self.merge_comprehension_result(result)
            self.last_comprehension_at = now
            self.cached_live_summary = build_live_summary(self.full_transcript, result)
            self.cached_live_summary_at = now

    def get_live_summary(self) -> str:
        """Return one paragraph describing the meeting so far."""
        if self.cached_live_summary:
            return self.cached_live_summary
        if not self.full_transcript:
            return "No transcript has been captured yet."
        self.cached_live_summary = build_live_summary(self.full_transcript, run_comprehension_pass(recent_segments(self.full_transcript, 600)))
        self.cached_live_summary_at = time.time()
        return self.cached_live_summary

    def log_agent_response(self, trigger: str, text: str, timestamp: float | None = None) -> None:
        """Record a response produced by the agent."""
        self.agent_responses_given.append(AgentResponseLog(timestamp or time.time(), trigger, text))

    def merge_comprehension_result(self, result: ComprehensionResult) -> None:
        """Merge a comprehension pass into runtime memory while deduplicating."""
        for decision in result.decisions:
            self._append_unique(self.decisions_detected, decision)
        for action_item in result.action_items:
            self._append_action_item(action_item)
        for question in result.open_questions:
            self._append_unique(self.questions_raised, question)
        for topic in result.topics:
            self._append_unique(self.topics_discussed, topic)
        for point in result.key_points:
            timestamp = self.full_transcript[-1].end if self.full_transcript else 0.0
            self._append_key_moment(KeyMoment(timestamp, point, "key_point", "medium"))

    def _append_unique(self, values: list[str], candidate: str, threshold: float = 0.8) -> None:
        cleaned = candidate.strip()
        if not cleaned:
            return
        for existing in values:
            if SequenceMatcher(None, existing.lower(), cleaned.lower()).ratio() >= threshold:
                return
        values.append(cleaned)

    def _append_action_item(self, candidate: ActionItem, threshold: float = 0.8) -> None:
        for existing in self.action_items_detected:
            if SequenceMatcher(None, existing.what.lower(), candidate.what.lower()).ratio() >= threshold:
                return
        self.action_items_detected.append(candidate)

    def _append_key_moment(self, candidate: KeyMoment, threshold: float = 0.8) -> None:
        for existing in self.key_moments:
            if existing.type == candidate.type and SequenceMatcher(None, existing.text.lower(), candidate.text.lower()).ratio() >= threshold:
                return
        self.key_moments.append(candidate)


class EmbeddingProvider:
    """Embeds text with sentence-transformers, OpenAI, or a deterministic local fallback."""

    def __init__(self) -> None:
        """Create an embedding provider using the best available configured backend."""
        if load_dotenv is not None:
            load_dotenv()
        self._sentence_model = None
        self._openai_client = None
        self._openai_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self._backend = "hash"
        self._load_sentence_transformer()
        if self._sentence_model is None:
            self._load_openai()

    def embed(self, text: str) -> list[float]:
        """Return an embedding vector for text, falling back locally on failure."""
        cleaned = text.strip()
        if not cleaned:
            return [0.0] * 384
        if self._sentence_model is not None:
            try:
                vector = self._sentence_model.encode(cleaned, normalize_embeddings=True)
                return [float(value) for value in vector.tolist()]
            except Exception as exc:
                logger.warning("Sentence-transformers embedding failed; using local fallback: %s", exc)
        if self._openai_client is not None:
            try:
                response = self._openai_client.embeddings.create(model=self._openai_model, input=cleaned)
                return [float(value) for value in response.data[0].embedding]
            except Exception as exc:
                logger.warning("OpenAI embedding failed; using local fallback: %s", exc)
        return _hash_embedding(cleaned)

    def _load_sentence_transformer(self) -> None:
        if os.getenv("ENABLE_LOCAL_EMBEDDINGS", "false").strip().lower() not in {"1", "true", "yes"}:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return
        try:
            model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            self._sentence_model = SentenceTransformer(model_name)
            self._backend = "sentence-transformers"
        except Exception as exc:
            logger.warning("Could not load local embedding model; trying other backends: %s", exc)
            self._sentence_model = None

    def _load_openai(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return
        try:
            from openai import OpenAI
        except ImportError:
            return
        self._openai_client = OpenAI(api_key=api_key)
        self._backend = "openai"


@dataclass
class SimpleVectorStore:
    """Small in-memory vector store used when Chroma is unavailable."""

    embedding_provider: EmbeddingProvider
    texts: list[str] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)

    def add_texts(self, texts: list[str]) -> None:
        """Add texts and their embeddings to the store."""
        for text in texts:
            cleaned = text.strip()
            if cleaned:
                self.texts.append(cleaned)
                self.embeddings.append(self.embedding_provider.embed(cleaned))

    def similarity_search(self, query: str, k: int = 3) -> list[str]:
        """Return the top-k stored chunks most similar to query."""
        if not self.texts:
            return []
        query_embedding = self.embedding_provider.embed(query)
        scored = [
            (cosine_similarity(query_embedding, embedding), text)
            for text, embedding in zip(self.texts, self.embeddings, strict=False)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [text for _, text in scored[:k]]


class ChromaVectorStore:
    """In-memory Chroma wrapper with a local fallback-compatible interface."""

    def __init__(self, embedding_provider: EmbeddingProvider, collection_name: str) -> None:
        """Create an ephemeral Chroma collection."""
        self.embedding_provider = embedding_provider
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise RuntimeError("chromadb is not installed") from exc
        self._client = chromadb.Client(Settings(anonymized_telemetry=False))
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def add_texts(self, texts: list[str]) -> None:
        """Add texts and explicit embeddings to Chroma."""
        cleaned_texts = [text.strip() for text in texts if text.strip()]
        if not cleaned_texts:
            return
        ids = [_stable_id(text, index) for index, text in enumerate(cleaned_texts)]
        embeddings = [self.embedding_provider.embed(text) for text in cleaned_texts]
        self._collection.add(ids=ids, documents=cleaned_texts, embeddings=embeddings)

    def similarity_search(self, query: str, k: int = 3) -> list[str]:
        """Return the top-k Chroma documents most similar to query."""
        if not query.strip():
            return []
        result = self._collection.query(query_embeddings=[self.embedding_provider.embed(query)], n_results=k)
        documents = result.get("documents") or [[]]
        return [str(item) for item in documents[0]]


def learn_from_briefing(briefing: Briefing) -> AgentMemory:
    """Convert a pre-meeting briefing into structured agent memory."""
    embedding_provider = EmbeddingProvider()
    system_prompt = build_system_prompt(briefing)
    avoid_embeddings = [
        EmbeddingRecord(text=topic, embedding=embedding_provider.embed(topic))
        for topic in briefing.topics_to_avoid
    ]
    chunks = chunk_background_context(briefing.background_context)
    vector_store = create_background_vector_store(embedding_provider, briefing.meeting_title)
    vector_store.add_texts(chunks)
    return AgentMemory(
        system_prompt=system_prompt,
        talking_points_queue=list(briefing.talking_points),
        raised_points_log=[],
        topics_to_avoid_embeddings=avoid_embeddings,
        background_vector_store=vector_store,
        briefing=briefing,
        embedding_provider=embedding_provider,
    )


def run_comprehension_pass(transcript_window: list[TranscriptSegment]) -> ComprehensionResult:
    """Extract decisions, actions, questions, topics, and quotes from a transcript window."""
    if not transcript_window:
        return ComprehensionResult([], [], [], [], "No conversation captured.", [], [])
    llm_result = _run_llm_comprehension(transcript_window)
    if llm_result is not None:
        return llm_result
    return _run_rule_based_comprehension(transcript_window)


def recent_segments(segments: list[TranscriptSegment], seconds: int) -> list[TranscriptSegment]:
    """Return transcript segments within the last N seconds."""
    if not segments:
        return []
    latest = max(segment.end for segment in segments)
    cutoff = latest - seconds
    return [segment for segment in segments if segment.end >= cutoff]


def build_live_summary(segments: list[TranscriptSegment], result: ComprehensionResult) -> str:
    """Build a concise live summary from transcript and comprehension state."""
    if not segments:
        return "No transcript has been captured yet."
    topics = ", ".join(result.topics[:3]) if result.topics else "the current agenda"
    decisions = f"{len(result.decisions)} decision(s)" if result.decisions else "no confirmed decisions yet"
    actions = f"{len(result.action_items)} action item(s)" if result.action_items else "no action items yet"
    latest_text = segments[-1].text.strip()
    return f"The meeting is discussing {topics}. There are {decisions} and {actions}. Most recently: {latest_text}"


def estimate_valence(text: str) -> float:
    """Estimate sentiment valence with a local lexical fallback."""
    positive_words = {"agree", "agreed", "great", "good", "clear", "confirmed", "progress", "thanks", "excellent", "resolved"}
    negative_words = {"concern", "blocked", "risk", "issue", "problem", "delay", "conflict", "disagree", "tension", "missed"}
    tokens = re.findall(r"[a-z']+", text.lower())
    if not tokens:
        return 0.0
    positive = sum(1 for token in tokens if token in positive_words)
    negative = sum(1 for token in tokens if token in negative_words)
    return max(-1.0, min(1.0, (positive - negative) / max(4, positive + negative + 1)))


def detect_decision_text(text: str) -> str | None:
    """Detect likely decision statements using deterministic phrases."""
    patterns = [
        r"\bwe(?:'| wi)?ll go with\b.+",
        r"\bagreed\b.+",
        r"\bdecided\b.+",
        r"\bthe plan is\b.+",
        r"\bconfirmed\b.+",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return None


def detect_action_item(text: str) -> ActionItem | None:
    """Detect likely action items using deterministic phrases."""
    patterns = [
        r"(?P<who>[A-Z][a-z]+)\s+will\s+(?P<what>.+)",
        r"will you\s+(?P<what>.+)",
        r"your action\s+is\s+(?P<what>.+)",
        r"take this forward\s*:?\s*(?P<what>.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            who = match.groupdict().get("who") or "unassigned"
            what = match.groupdict().get("what") or match.group(0)
            by_when = _extract_due_date(text)
            return ActionItem(what=what.strip(), who=who.strip(), by_when=by_when, confidence=0.65)
    if re.search(r"\bby\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|next week|\d{1,2}/\d{1,2})\b", text, re.IGNORECASE):
        return ActionItem(what=text.strip(), who="unassigned", by_when=_extract_due_date(text), confidence=0.55)
    return None


def _run_llm_comprehension(transcript_window: list[TranscriptSegment]) -> ComprehensionResult | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    transcript_text = "\n".join(f"{segment.speaker_label}: {segment.text}" for segment in transcript_window)
    prompt = (
        "You are analysing a meeting transcript segment. Extract JSON with keys: "
        "key_points, decisions, action_items, open_questions, sentiment_summary, notable_quotes, topics. "
        "action_items must be objects with what, who, by_when, confidence.\n\n"
        f"{transcript_text}"
    )
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MEMORY_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
            messages=[
                {"role": "system", "content": "Return only valid JSON for meeting comprehension."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "20")),
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        return _comprehension_from_mapping(payload)
    except Exception as exc:
        logger.warning("LLM comprehension failed; using rule-based fallback: %s", exc)
        return None


def _run_rule_based_comprehension(transcript_window: list[TranscriptSegment]) -> ComprehensionResult:
    decisions: list[str] = []
    action_items: list[ActionItem] = []
    open_questions: list[str] = []
    notable_quotes: list[str] = []
    topics: list[str] = []
    for segment in transcript_window:
        decision = detect_decision_text(segment.text)
        if decision and not _contains_similar(decisions, decision):
            decisions.append(decision)
        action_item = detect_action_item(segment.text)
        if action_item is not None and not _contains_similar([item.what for item in action_items], action_item.what):
            action_items.append(action_item)
        if "?" in segment.text and not _contains_similar(open_questions, segment.text):
            open_questions.append(segment.text.strip())
        if len(segment.text.split()) >= 12 and len(notable_quotes) < 3:
            notable_quotes.append(segment.text.strip())
        topic = _guess_topic(segment.text)
        if topic and not _contains_similar(topics, topic):
            topics.append(topic)
    key_points = [segment.text.strip() for segment in transcript_window if len(segment.text.split()) >= 8][:5]
    sentiment = _summarize_sentiment([estimate_valence(segment.text) for segment in transcript_window])
    return ComprehensionResult(key_points[:5], decisions, action_items, open_questions, sentiment, notable_quotes, topics[:5])


def _comprehension_from_mapping(payload: dict[str, object]) -> ComprehensionResult:
    action_items: list[ActionItem] = []
    for item in _coerce_list(payload.get("action_items")):
        if isinstance(item, dict):
            action_items.append(
                ActionItem(
                    what=str(item.get("what") or ""),
                    who=str(item.get("who") or "unassigned"),
                    by_when=str(item.get("by_when")) if item.get("by_when") else None,
                    confidence=float(item.get("confidence") or 0.5),
                )
            )
        elif str(item).strip():
            action_items.append(ActionItem(str(item), "unassigned", None, 0.4))
    return ComprehensionResult(
        key_points=[str(item) for item in _coerce_list(payload.get("key_points")) if str(item).strip()],
        decisions=[str(item) for item in _coerce_list(payload.get("decisions")) if str(item).strip()],
        action_items=action_items,
        open_questions=[str(item) for item in _coerce_list(payload.get("open_questions")) if str(item).strip()],
        sentiment_summary=str(payload.get("sentiment_summary") or "No sentiment summary available."),
        notable_quotes=[str(item) for item in _coerce_list(payload.get("notable_quotes")) if str(item).strip()],
        topics=[str(item) for item in _coerce_list(payload.get("topics")) if str(item).strip()],
    )


def build_system_prompt(briefing: Briefing) -> str:
    """Build the durable system prompt the agent consults during the meeting."""
    mode_guidance = {
        AgentMode.FULL_PROXY: "Act as a transparent proxy for the user, speaking from their authorized perspective.",
        AgentMode.SILENT_OBSERVER: "Never speak aloud; record, analyze, and preserve meeting context only.",
        AgentMode.ADVISOR: "Provide private text-only suggestions to the user; never speak aloud.",
        AgentMode.PARTICIPATOR: "Join naturally when useful and contribute relevant points without dominating.",
    }[briefing.agent_mode]
    lines = [
        "You are an AI Meeting Assistant configured from a pre-meeting briefing.",
        f"Meeting title: {briefing.meeting_title}",
        f"Objective: {briefing.meeting_objective}",
        f"Known attendees: {_format_list(briefing.attendees)}",
        f"User name: {briefing.user_name or 'Not provided'}",
        f"Assistant display name: {briefing.assistant_display_name or 'Not provided'}",
        f"User role: {briefing.user_role}",
        f"Agent mode: {briefing.agent_mode.value}",
        f"Mode guidance: {mode_guidance}",
        f"Response style: {briefing.response_style}",
        f"Speaking persona: {briefing.speaking_persona or 'Use the response style and sound natural.'}",
        f"Decision authority: {briefing.decision_authority}",
        f"Escalation rules: {_format_list(briefing.escalation_rules)}",
        f"Auto respond: {briefing.auto_respond}",
        f"Maximum speaking turns: {briefing.max_speaking_turns}",
        f"Talking points to raise when relevant: {_format_list(briefing.talking_points)}",
        f"Topics to avoid: {_format_list(briefing.topics_to_avoid)}",
        f"Custom instructions: {_format_list(briefing.custom_instructions)}",
        "Before every response, check semantic similarity against topics to avoid.",
        "If the current topic is too similar to an avoided topic, stay silent or deflect briefly.",
        "When in FULL_PROXY mode, you are representing the user as a disclosed assistant proxy.",
        "In FULL_PROXY mode, speak in first person, match the user's persona, and stay inside the stated decision authority.",
        "If a question exceeds the user's decision authority or matches an escalation rule, defer clearly instead of inventing a commitment.",
        "Do not claim to be the human user. If identity is relevant, be transparent that you are the user's AI assistant.",
        "Speak naturally, use first person for authorized project experience when appropriate, and match the user's role and style.",
        f"Background context: {briefing.background_context or 'No additional background provided.'}",
    ]
    return "\n".join(lines)


def chunk_background_context(context: str, max_chars: int = 700, overlap_chars: int = 120) -> list[str]:
    """Split briefing background context into overlapping chunks for retrieval."""
    cleaned = re.sub(r"\s+", " ", context.strip())
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + max_chars)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(cleaned):
            break
        start = max(0, end - overlap_chars)
    return chunks


def create_background_vector_store(embedding_provider: EmbeddingProvider, meeting_title: str) -> VectorStore:
    """Create an in-memory Chroma collection, falling back to a simple vector store."""
    if os.getenv("ENABLE_CHROMA_VECTOR_STORE", "false").strip().lower() not in {"1", "true", "yes"}:
        return SimpleVectorStore(embedding_provider)
    collection_name = "meeting_background_" + re.sub(r"[^a-zA-Z0-9_]+", "_", meeting_title.lower()).strip("_")
    collection_name = collection_name[:60] or "meeting_background"
    try:
        return ChromaVectorStore(embedding_provider, collection_name)
    except Exception as exc:
        logger.info("Using simple in-memory vector store instead of Chroma: %s", exc)
        return SimpleVectorStore(embedding_provider)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not left or not right:
        return 0.0
    length = min(len(left), len(right))
    dot = sum(left[index] * right[index] for index in range(length))
    left_norm = math.sqrt(sum(left[index] * left[index] for index in range(length)))
    right_norm = math.sqrt(sum(right[index] * right[index] for index in range(length)))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _hash_embedding(text: str, dimensions: int = 384) -> list[float]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    vector = [0.0] * dimensions
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _stable_id(text: str, index: int) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"chunk_{index}_{digest[:16]}"


def _format_list(items: list[str]) -> str:
    return ", ".join(items) if items else "None provided"


def _find_lexical_topic_match(text: str, topics: list[str]) -> str | None:
    normalized_text = " " + re.sub(r"\s+", " ", text.lower()).strip() + " "
    text_tokens = set(re.findall(r"[a-z0-9]+", normalized_text))
    for topic in topics:
        normalized_topic = " " + re.sub(r"\s+", " ", topic.lower()).strip() + " "
        if normalized_topic.strip() and normalized_topic in normalized_text:
            return topic
        topic_tokens = set(re.findall(r"[a-z0-9]+", normalized_topic))
        if topic_tokens and len(topic_tokens & text_tokens) / len(topic_tokens) >= 0.8:
            return topic
    return None


def _extract_due_date(text: str) -> str | None:
    match = re.search(
        r"\bby\s+(?P<due>monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|next week|\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b",
        text,
        flags=re.IGNORECASE,
    )
    return match.group("due") if match else None


def _contains_similar(values: list[str], candidate: str, threshold: float = 0.8) -> bool:
    return any(SequenceMatcher(None, value.lower(), candidate.lower()).ratio() >= threshold for value in values)


def _guess_topic(text: str) -> str:
    tokens = [token for token in re.findall(r"[a-z][a-z0-9-]+", text.lower()) if len(token) > 4]
    stopwords = {"about", "there", "their", "which", "should", "would", "could", "meeting", "think", "going"}
    candidates = [token for token in tokens if token not in stopwords]
    return " ".join(candidates[:3]) if candidates else ""


def _summarize_sentiment(values: list[float]) -> str:
    if not values:
        return "No emotional tone detected yet."
    average = sum(values) / len(values)
    if average > 0.2:
        return "The tone is generally positive and constructive."
    if average < -0.2:
        return "The tone is strained and may need careful framing."
    return "The tone is mostly neutral."


def _coerce_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def runtime_memory_to_dict(memory: RuntimeMemory) -> dict[str, object]:
    """Return RuntimeMemory as JSON-serializable primitives."""
    data = asdict(memory)
    data["full_transcript"] = [segment.to_dict() for segment in memory.full_transcript]
    return data
