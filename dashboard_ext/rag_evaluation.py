# dashboard_ext/rag_evaluation.py
"""Lightweight RAG evaluation metrics for dashboard question answering."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class RagEvaluation:
    """Evaluation metrics for one RAG question-answering turn."""

    hallucination_percentage: float
    faithfulness_percentage: float
    meeting_relevance_percentage: float
    evaluation_reason: str
    query_coverage: float
    mean_document_relevance: float
    max_document_relevance: float
    citation_coverage: float
    retrieved_meeting_count: int
    retrieved_segment_count: int
    faithfulness_score: float | None
    faithfulness_reason: str
    document_scores: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return metrics as a plain dictionary."""
        return {
            "query_coverage": self.query_coverage,
            "hallucination_percentage": self.hallucination_percentage,
            "faithfulness_percentage": self.faithfulness_percentage,
            "meeting_relevance_percentage": self.meeting_relevance_percentage,
            "evaluation_reason": self.evaluation_reason,
            "mean_document_relevance": self.mean_document_relevance,
            "max_document_relevance": self.max_document_relevance,
            "citation_coverage": self.citation_coverage,
            "retrieved_meeting_count": self.retrieved_meeting_count,
            "retrieved_segment_count": self.retrieved_segment_count,
            "faithfulness_score": self.faithfulness_score,
            "faithfulness_reason": self.faithfulness_reason,
            "document_scores": self.document_scores,
        }


def evaluate_rag_turn(
    question: str,
    answer: str,
    docs: list[object],
    llm: object | None = None,
    original_meeting_text: str = "",
) -> RagEvaluation:
    """Evaluate retrieval relevance and optional answer faithfulness for one RAG turn."""
    question_terms = _content_terms(question)
    answer_terms = _content_terms(answer)
    original_meeting_terms = set(_content_terms(original_meeting_text))
    document_scores: list[dict[str, Any]] = []
    retrieved_terms: set[str] = set()
    meeting_ids: set[str] = set()
    segment_ids: set[str] = set()
    cited_segment_count = 0

    for index, doc in enumerate(docs, start=1):
        text = getattr(doc, "page_content", "") or ""
        metadata = getattr(doc, "metadata", {}) or {}
        doc_terms = set(_content_terms(text))
        retrieved_terms.update(doc_terms)
        meeting_id = str(metadata.get("meeting_id", ""))
        segment = str(metadata.get("segment", ""))
        if meeting_id:
            meeting_ids.add(meeting_id)
        if segment:
            segment_ids.add(f"{meeting_id}:{segment}")
        relevance = _overlap_score(question_terms, doc_terms)
        if _answer_cites(answer, meeting_id, segment):
            cited_segment_count += 1
        document_scores.append(
            {
                "rank": index,
                "meeting_id": meeting_id,
                "segment": segment,
                "split": metadata.get("split"),
                "relevance": relevance,
                "matched_terms": sorted(set(question_terms) & doc_terms),
            }
        )

    relevance_values = [item["relevance"] for item in document_scores]
    query_coverage = _overlap_score(question_terms, retrieved_terms)
    citation_coverage = cited_segment_count / len(docs) if docs else 0.0
    fallback_faithfulness = _overlap_score(answer_terms, retrieved_terms)
    fallback_meeting_relevance = _overlap_score(answer_terms, original_meeting_terms or retrieved_terms)
    judged = _judge_answer_quality(question, answer, docs, original_meeting_text, llm)
    if judged is None:
        faithfulness_percentage = fallback_faithfulness * 100.0
        meeting_relevance_percentage = fallback_meeting_relevance * 100.0
        hallucination_percentage = max(0.0, 100.0 - faithfulness_percentage)
        evaluation_reason = (
            "Lexical fallback: hallucination is estimated from answer terms not found in retrieved context; "
            "meeting relevance is estimated from answer terms found in the original meeting transcript."
        )
        faithfulness_score = None
        faithfulness_reason = "LLM judge unavailable; using lexical fallback percentages."
    else:
        hallucination_percentage = judged["hallucination_percentage"]
        faithfulness_percentage = judged["faithfulness_percentage"]
        meeting_relevance_percentage = judged["meeting_relevance_percentage"]
        evaluation_reason = judged["reason"]
        faithfulness_score = faithfulness_percentage / 100.0
        faithfulness_reason = judged["reason"]
    return RagEvaluation(
        hallucination_percentage=hallucination_percentage,
        faithfulness_percentage=faithfulness_percentage,
        meeting_relevance_percentage=meeting_relevance_percentage,
        evaluation_reason=evaluation_reason,
        query_coverage=query_coverage,
        mean_document_relevance=sum(relevance_values) / len(relevance_values) if relevance_values else 0.0,
        max_document_relevance=max(relevance_values) if relevance_values else 0.0,
        citation_coverage=citation_coverage,
        retrieved_meeting_count=len(meeting_ids),
        retrieved_segment_count=len(segment_ids),
        faithfulness_score=faithfulness_score,
        faithfulness_reason=faithfulness_reason,
        document_scores=document_scores,
    )


def _judge_answer_quality(
    question: str,
    answer: str,
    docs: list[object],
    original_meeting_text: str,
    llm: object | None,
) -> dict[str, Any] | None:
    if llm is None:
        return None
    context = "\n\n".join(getattr(doc, "page_content", "") or "" for doc in docs)
    prompt = {
        "question": question,
        "answer": answer,
        "retrieved_context": context[:12000],
        "original_meeting_transcript": original_meeting_text[:12000],
        "instruction": (
            "Evaluate the answer as a RAG judge. Return JSON with: "
            "hallucination_percentage, faithfulness_percentage, meeting_relevance_percentage, reason. "
            "Hallucination is the percent of answer content unsupported by retrieved context. "
            "Faithfulness is the percent of answer claims supported by retrieved context. "
            "Meeting relevance is the percent of answer content relevant to the original meeting transcript. "
            "Each percentage must be from 0 to 100."
        ),
    }
    try:
        response = llm.invoke([
            ("system", "You are a strict RAG evaluator. Return only valid JSON."),
            ("user", json.dumps(prompt)),
        ])
        payload = _extract_json(getattr(response, "content", str(response)))
        return {
            "hallucination_percentage": _bounded_percentage(payload.get("hallucination_percentage", 0.0)),
            "faithfulness_percentage": _bounded_percentage(payload.get("faithfulness_percentage", 0.0)),
            "meeting_relevance_percentage": _bounded_percentage(payload.get("meeting_relevance_percentage", 0.0)),
            "reason": str(payload.get("reason", "No reason provided.")),
        }
    except Exception:
        return None


def _content_terms(text: str) -> list[str]:
    stopwords = {
        "about",
        "after",
        "again",
        "also",
        "and",
        "are",
        "from",
        "have",
        "meeting",
        "segment",
        "that",
        "the",
        "their",
        "there",
        "this",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "with",
    }
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 3 and token not in stopwords
    ]


def _overlap_score(left_terms: list[str] | set[str], right_terms: set[str]) -> float:
    left = set(left_terms)
    if not left:
        return 0.0
    return len(left & right_terms) / len(left)


def _answer_cites(answer: str, meeting_id: str, segment: str) -> bool:
    lowered = answer.lower()
    return bool(
        (meeting_id and meeting_id.lower() in lowered)
        or (segment and re.search(rf"\bsegment\s+{re.escape(segment)}\b", lowered))
    )


def _extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _bounded_percentage(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(100.0, numeric))
