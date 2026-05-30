import ast
import json
import os
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        RecursiveCharacterTextSplitter = None

try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.docstore.document import Document
    except ImportError:
        Document = None

try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
except ImportError:
    try:
        from langchain.embeddings import OpenAIEmbeddings
        from langchain.chat_models import ChatOpenAI
    except ImportError:
        OpenAIEmbeddings = None
        ChatOpenAI = None

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    try:
        from langchain.vectorstores import FAISS
    except ImportError:
        FAISS = None


st.set_page_config(page_title="Meeting Health Dashboard", layout="wide")

APP_DIR = Path(__file__).resolve().parent
LOCAL_PROJECT_DIR = APP_DIR.parent
if str(LOCAL_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_PROJECT_DIR))

try:
    from dashboard_ext.assistant_tab import render_assistant_tab
except ImportError:
    render_assistant_tab = None

PROJECT_DIR = Path("/Users/saiteja/Desktop/meeting-intelligence-platform")
NOTEBOOK_DIR = PROJECT_DIR / "notebooks"
SEARCH_DIRS = [
    APP_DIR,
    APP_DIR / "data",
    LOCAL_PROJECT_DIR,
    LOCAL_PROJECT_DIR / "data",
    LOCAL_PROJECT_DIR / "notebooks",
    LOCAL_PROJECT_DIR / "notebooks" / "data",
    Path.cwd(),
    Path.cwd() / "data",
    PROJECT_DIR,
    PROJECT_DIR / "data",
    NOTEBOOK_DIR,
    NOTEBOOK_DIR / "data",
]

DATA_FILES = {
    "metrics": "meeting_metrics_df.csv",
    "chunks": "chunk_level_df.csv",
    "moments": "key_moments_df.csv",
    "topics": "topic_health.csv",
    "summaries": "meeting_summaries_df.csv",
}


st.title("Meeting Health Dashboard")
st.write(
    "Monitor meeting quality, summarize important outcomes, spot conversation risks, "
    "and ask questions across meeting history from one workspace."
)


# ----------------- LOADERS -----------------
@st.cache_data
def load_csv_from_path(path: str) -> pd.DataFrame:
    return clean_frame(pd.read_csv(path))


@st.cache_data
def load_csv_from_upload(file) -> pd.DataFrame:
    return clean_frame(pd.read_csv(file))


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def find_local_file(filename: str) -> Path | None:
    candidates = []
    for folder in SEARCH_DIRS:
        candidate = folder / filename
        if candidate.exists():
            candidates.append(candidate)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_dataset(name: str, upload_file=None):
    filename = DATA_FILES[name]
    if upload_file is not None:
        return load_csv_from_upload(upload_file), "Manual override"

    path = find_local_file(filename)
    if path is None:
        return None, "Not found"
    return load_csv_from_path(str(path)), f"Loaded automatically from {path.parent}"


def get_recordings_dir() -> Path:
    load_private_env()
    configured = Path(os.getenv("RECORDINGS_DIR", "./recordings")).expanduser()
    return configured if configured.is_absolute() else LOCAL_PROJECT_DIR / configured


def list_assistant_sessions() -> list[str]:
    recordings_dir = get_recordings_dir()
    if not recordings_dir.exists():
        return []
    sessions = []
    for folder in recordings_dir.iterdir():
        if not folder.is_dir():
            continue
        if (folder / "pipeline").exists() or (folder / "session_metadata.json").exists():
            sessions.append(folder.name)
    return sorted(sessions, reverse=True)


def assistant_session_dir(session_id: str | None) -> Path | None:
    if not session_id:
        return None
    path = get_recordings_dir() / session_id
    return path if path.exists() else None


def load_assistant_dataset(session_dir: Path, name: str):
    path = session_dir / "pipeline" / DATA_FILES[name]
    if not path.exists():
        return None, f"Missing from assistant session: {path}"
    return load_csv_from_path(str(path)), f"Loaded from assistant session {session_dir.name}"


@st.cache_data
def compute_meeting_metrics_from_chunks(chunk_df: pd.DataFrame) -> pd.DataFrame:
    """Fallback only: derive basic meeting metrics from available conversation flow data."""
    if "valence" not in chunk_df.columns:
        raise ValueError("Conversation data must include a valence column.")

    grouped = chunk_df.groupby("meeting_id")["valence"]
    avg_val = grouped.mean()
    std_val = grouped.std().fillna(0.0)
    health_score = ((avg_val + 1) / 2 * 100).clip(0, 100)
    max_std = std_val.max() if std_val.max() > 0 else 1.0
    collab = (1 - std_val / max_std).clip(0, 1) * 100
    neg_ratio = (
        chunk_df.assign(is_neg=chunk_df["valence"] < -0.2)
        .groupby("meeting_id")["is_neg"].mean() * 100
    )

    return pd.DataFrame({
        "meeting_id": avg_val.index,
        "meeting_health_valence": avg_val.values,
        "meeting_health_score": health_score.values,
        "collaboration_score": collab.reindex(avg_val.index).values,
        "tension_score": neg_ratio.reindex(avg_val.index).fillna(0).values,
        "predicted_health_label": "estimated_from_conversation_flow",
    })


def find_key_shifts(chunks_meeting: pd.DataFrame, threshold: float = 0.4, top_n: int = 5) -> pd.DataFrame:
    if "valence" not in chunks_meeting.columns or "chunk_idx" not in chunks_meeting.columns:
        return pd.DataFrame(columns=["chunk_idx", "delta_valence", "chunk_text"])

    cm = chunks_meeting.sort_values("chunk_idx").reset_index(drop=True)
    if len(cm) < 2:
        return pd.DataFrame(columns=["chunk_idx", "delta_valence", "chunk_text"])

    if "delta_valence" not in cm.columns:
        cm["prev_valence"] = cm["valence"].shift(1)
        cm["delta_valence"] = cm["valence"] - cm["prev_valence"]

    cm["abs_delta"] = cm["delta_valence"].abs()
    return cm[cm["abs_delta"] >= threshold].sort_values("abs_delta", ascending=False).head(top_n)[
        ["chunk_idx", "delta_valence", "chunk_text"]
    ]


def parse_structured_field(value):
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if not isinstance(value, str):
        return [value]
    value = value.strip()
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else [parsed]
    except Exception:
        return [value]


def make_openai_embeddings(model_name: str, api_key: str):
    try:
        return OpenAIEmbeddings(model=model_name, api_key=api_key)
    except TypeError:
        return OpenAIEmbeddings(model=model_name, openai_api_key=api_key)


def make_chat_openai(model_name: str, api_key: str):
    try:
        return ChatOpenAI(model=model_name, temperature=0.1, api_key=api_key)
    except TypeError:
        return ChatOpenAI(model_name=model_name, temperature=0.1, openai_api_key=api_key)


def load_private_env() -> None:
    """Load private local environment values without showing them in the dashboard."""
    for env_path in [LOCAL_PROJECT_DIR / ".env", APP_DIR / ".env", Path.cwd() / ".env"]:
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def get_service_key() -> str:
    load_private_env()
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        secret_key = ""
    return secret_key or os.getenv("OPENAI_API_KEY", "")


def display_label(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return str(value).replace("_", " ").title()


def display_item(item):
    if isinstance(item, dict):
        text = item.get("task") or item.get("decision") or item.get("point") or item.get("description") or str(item)
        owner = item.get("owner")
        due_date = item.get("due_date")
        details = []
        if owner and owner != "unknown":
            details.append(f"Owner: {owner}")
        if due_date and str(due_date).lower() not in {"none", "null", "nan"}:
            details.append(f"Due: {due_date}")
        return f"{text} ({'; '.join(details)})" if details else text
    return str(item)


SUMMARY_PLACEHOLDER_TEXT = "OPENAI_API_KEY not set"


def summary_is_placeholder(row) -> bool:
    if row is None:
        return True
    summary = str(row.get("summary", ""))
    return not summary.strip() or SUMMARY_PLACEHOLDER_TEXT in summary


def extract_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def meeting_text_from_chunks(meeting_id: str, chunk_df: pd.DataFrame) -> str:
    meeting_chunks = (
        chunk_df[chunk_df["meeting_id"] == meeting_id]
        .sort_values("chunk_idx")
    )
    if "chunk_text" not in meeting_chunks.columns:
        return ""
    return "\n".join(str(text) for text in meeting_chunks["chunk_text"].dropna().tolist())


def generate_summary_from_meeting_text(meeting_id: str, transcript: str, api_key: str) -> dict:
    if not api_key:
        raise RuntimeError("The summary service is not configured for this environment.")
    if ChatOpenAI is None:
        raise RuntimeError("The summary service dependencies are not installed.")

    schema = {
        "summary": "2-4 sentence summary",
        "action_items": [{"owner": "owner if stated, otherwise unknown", "task": "specific action", "due_date": "date if stated, otherwise null"}],
        "key_decisions": ["decision"],
        "tension_points": ["risk, disagreement, blocker, or confusion"],
    }
    prompt = f"""
Return only valid JSON using this schema:
{json.dumps(schema, indent=2)}

Meeting ID: {meeting_id}
Transcript:
{transcript[:12000]}
""".strip()
    llm = make_chat_openai(os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini"), api_key)
    response = llm.invoke([
        ("system", "You extract concise meeting outcomes from transcripts."),
        ("user", prompt),
    ])
    parsed = extract_json_object(response.content)
    parsed["meeting_id"] = meeting_id
    parsed["model"] = os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini")
    return parsed


def upsert_summary_row(summary_df: pd.DataFrame | None, summary: dict) -> pd.DataFrame:
    row_df = pd.DataFrame([summary])
    if summary_df is None or summary_df.empty or "meeting_id" not in summary_df.columns:
        return row_df
    remaining = summary_df[summary_df["meeting_id"] != summary["meeting_id"]]
    return pd.concat([remaining, row_df], ignore_index=True)


def persist_summaries(summary_df: pd.DataFrame) -> None:
    data_dir = APP_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(data_dir / "meeting_summaries_df.csv", index=False)
    records = summary_df.to_dict(orient="records")
    (data_dir / "meeting_summaries.json").write_text(json.dumps(records, indent=2))


@st.cache_resource(show_spinner=False)
def build_faiss_store(records, api_key: str, embedding_model_name: str):
    if not all([RecursiveCharacterTextSplitter, Document, OpenAIEmbeddings, FAISS]):
        raise RuntimeError("The question-answering service is not fully configured in this environment.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents = []
    for meeting_id, chunk_idx, chunk_text in records:
        for split_idx, text in enumerate(splitter.split_text(chunk_text or "")):
            documents.append(Document(
                page_content=text,
                metadata={"meeting_id": meeting_id, "segment": chunk_idx, "split": split_idx},
            ))

    embeddings = make_openai_embeddings(embedding_model_name, api_key)
    return FAISS.from_documents(documents, embeddings)


# ----------------- DATA SOURCES -----------------
st.sidebar.header("Data Status")
st.sidebar.caption("The dashboard loads prepared files automatically. Manual files are optional overrides.")

with st.sidebar.expander("Optional manual overrides"):
    uploaded_files = {
        "metrics": st.file_uploader("Meeting scores", type=["csv"]),
        "chunks": st.file_uploader("Conversation segments", type=["csv"]),
        "moments": st.file_uploader("Important moments", type=["csv"]),
        "topics": st.file_uploader("Topic insights", type=["csv"]),
        "summaries": st.file_uploader("Meeting summaries", type=["csv"]),
    }

assistant_sessions = list_assistant_sessions()
selected_assistant_session = None
selected_assistant_session_dir = None
st.sidebar.subheader("Load from Assistant Session")
if assistant_sessions:
    selected_assistant_session = st.sidebar.selectbox(
        "Assistant session",
        [""] + assistant_sessions,
        format_func=lambda value: "Use prepared/manual files" if value == "" else value,
    )
    selected_assistant_session_dir = assistant_session_dir(selected_assistant_session)
    if selected_assistant_session_dir is not None:
        st.sidebar.caption(f"Using {selected_assistant_session_dir}")
else:
    st.sidebar.caption("No assistant sessions found yet.")

dataframes = {}
statuses = {}
for name in DATA_FILES:
    dataframes[name], statuses[name] = load_dataset(name, uploaded_files.get(name))

if selected_assistant_session_dir is not None:
    for name in ["metrics", "chunks", "moments", "topics"]:
        assistant_df, assistant_status = load_assistant_dataset(selected_assistant_session_dir, name)
        if assistant_df is not None:
            dataframes[name] = assistant_df
            statuses[name] = assistant_status
        else:
            st.sidebar.warning(f"{DATA_FILES[name]} missing for selected assistant session")

for label, name in [
    ("Meeting scores", "metrics"),
    ("Conversation segments", "chunks"),
    ("Important moments", "moments"),
    ("Topic insights", "topics"),
    ("Meeting summaries", "summaries"),
]:
    if dataframes[name] is not None:
        st.sidebar.success(f"{label}: ready")
    else:
        st.sidebar.warning(f"{label}: waiting")

meeting_metrics_df = dataframes["metrics"]
chunk_level_df = dataframes["chunks"]
key_moments_df = dataframes["moments"]
topic_health = dataframes["topics"]
meeting_summaries_df = dataframes["summaries"]

if chunk_level_df is None:
    st.info(
        "No prepared meeting data was found yet. Run the notebook pipeline once, and the dashboard "
        "will load the generated files automatically on refresh."
    )
    with st.expander("Where the dashboard looks for prepared files"):
        st.write([str(folder) for folder in SEARCH_DIRS])
    st.stop()

required_chunk_cols = {"meeting_id", "chunk_idx", "valence"}
missing_chunk_cols = required_chunk_cols - set(chunk_level_df.columns)
if missing_chunk_cols:
    st.error(f"The conversation segments file is missing required columns: {missing_chunk_cols}")
    st.stop()

if meeting_metrics_df is None:
    meeting_metrics_df = compute_meeting_metrics_from_chunks(chunk_level_df)

if key_moments_df is not None:
    required_moment_cols = {"meeting_id", "chunk_idx", "delta_valence"}
    if required_moment_cols - set(key_moments_df.columns):
        key_moments_df = None


# ----------------- DASHBOARD -----------------
meeting_ids = meeting_metrics_df["meeting_id"].dropna().unique().tolist()
selected_meeting = st.sidebar.selectbox("Meeting", meeting_ids)

m_row = meeting_metrics_df[meeting_metrics_df["meeting_id"] == selected_meeting].iloc[0]
chunks_meeting = chunk_level_df[chunk_level_df["meeting_id"] == selected_meeting].copy().sort_values("chunk_idx")

st.subheader(f"Meeting: {selected_meeting}")
score_cols = st.columns(5)
score_cols[0].metric("Health Score", f"{m_row['meeting_health_score']:.1f}")
score_cols[1].metric("Collaboration", f"{m_row['collaboration_score']:.1f}")
score_cols[2].metric("Tension", f"{m_row['tension_score']:.1f}")
score_cols[3].metric("Overall Tone", f"{m_row.get('meeting_health_valence', np.nan):.2f}" if "meeting_health_valence" in m_row.index else "N/A")
score_cols[4].metric("Status", display_label(m_row.get("predicted_health_label", "N/A")))

overview_tab, detail_tab, topics_tab, ask_tab, assistant_session_tab, source_tab = st.tabs([
    "Overview", "Meeting Details", "Topics", "Ask", "Assistant Session", "Source Data"
])

with overview_tab:
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Health Across Meetings")
        chart_cols = [c for c in ["meeting_health_score", "collaboration_score", "tension_score"] if c in meeting_metrics_df.columns]
        if chart_cols:
            st.bar_chart(meeting_metrics_df.set_index("meeting_id")[chart_cols])
    with right:
        st.subheader("Meeting Summary")
        api_key = get_service_key()
        summary_row = None
        if meeting_summaries_df is not None and "meeting_id" in meeting_summaries_df.columns:
            summary_rows = meeting_summaries_df[meeting_summaries_df["meeting_id"] == selected_meeting]
            if not summary_rows.empty:
                summary_row = summary_rows.iloc[0]

        if summary_is_placeholder(summary_row) and api_key:
            with st.spinner("Preparing meeting summary..."):
                transcript = meeting_text_from_chunks(selected_meeting, chunk_level_df)
                if transcript:
                    generated = generate_summary_from_meeting_text(selected_meeting, transcript, api_key)
                    meeting_summaries_df = upsert_summary_row(meeting_summaries_df, generated)
                    persist_summaries(meeting_summaries_df)
                    summary_row = pd.Series(generated)

        if summary_row is not None and not summary_is_placeholder(summary_row):
            st.write(summary_row.get("summary", "No summary available."))
        elif not api_key:
            st.info("Meeting summaries are waiting for the private service key in the app environment.")
        else:
            st.info("No meeting text was available to prepare a summary.")

    st.subheader("Conversation Tone Over Time")
    if not chunks_meeting.empty:
        st.line_chart(chunks_meeting.set_index("chunk_idx")[["valence"]])
    else:
        st.info("No conversation segments found for this meeting.")

with detail_tab:
    st.subheader("Outcomes and Risks")
    if meeting_summaries_df is not None and "meeting_id" in meeting_summaries_df.columns:
        summary_rows = meeting_summaries_df[meeting_summaries_df["meeting_id"] == selected_meeting]
        if not summary_rows.empty and not summary_is_placeholder(summary_rows.iloc[0]):
            s_row = summary_rows.iloc[0]
            a_col, d_col, t_col = st.columns(3)
            with a_col:
                st.markdown("**Action Items**")
                items = parse_structured_field(s_row.get("action_items"))
                if items:
                    for item in items:
                        st.write(f"- {display_item(item)}")
                else:
                    st.caption("None captured")
            with d_col:
                st.markdown("**Key Decisions**")
                items = parse_structured_field(s_row.get("key_decisions"))
                if items:
                    for item in items:
                        st.write(f"- {display_item(item)}")
                else:
                    st.caption("None captured")
            with t_col:
                st.markdown("**Tension Points**")
                items = parse_structured_field(s_row.get("tension_points"))
                if items:
                    for item in items:
                        st.write(f"- {display_item(item)}")
                else:
                    st.caption("None captured")
        else:
            st.info("No prepared outcomes found for this meeting.")
    else:
        st.info("Prepared outcomes will appear here after the pipeline runs.")

    st.subheader("Important Moments")
    if key_moments_df is not None:
        km_meeting = key_moments_df[key_moments_df["meeting_id"] == selected_meeting]
    else:
        km_meeting = find_key_shifts(chunks_meeting, threshold=0.4, top_n=5)

    if not km_meeting.empty:
        sort_cols = [c for c in ["shift_rank", "abs_delta_valence"] if c in km_meeting.columns]
        if sort_cols:
            km_meeting = km_meeting.sort_values(sort_cols, ascending=[True] * len(sort_cols))
        for _, row in km_meeting.iterrows():
            st.markdown(f"**Segment {int(row['chunk_idx'])}** - tone change `{row['delta_valence']:.2f}`")
            if "chunk_text" in km_meeting.columns and isinstance(row.get("chunk_text"), str):
                st.write(row["chunk_text"])
            st.divider()
    else:
        st.write("No major tone changes detected above the threshold.")

with topics_tab:
    st.subheader("Topics and Meeting Health")
    if topic_health is not None:
        if "topic" in topic_health.columns:
            display_cols = [c for c in ["topic", "keywords", "avg_health_score", "avg_meeting_health", "meeting_count"] if c in topic_health.columns]
            topic_display = topic_health[display_cols].rename(columns={
                "topic": "Topic",
                "keywords": "Keywords",
                "avg_health_score": "Average Health Score",
                "avg_meeting_health": "Average Tone",
                "meeting_count": "Meeting Count",
            })
            st.dataframe(topic_display.set_index("Topic"), use_container_width=True)
            if "Average Health Score" in topic_display.columns:
                st.bar_chart(topic_display.set_index("Topic")[["Average Health Score"]])
        else:
            st.dataframe(topic_health, use_container_width=True)
    else:
        st.info("Topic insights will appear here after the preparation pipeline runs.")

with ask_tab:
    st.subheader("Ask About Meetings")
    api_key = get_service_key()
    embedding_model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    answer_model_name = os.getenv("OPENAI_RAG_MODEL", "gpt-4o-mini")
    question = st.text_input("Question", value="What were the key decisions, action items, and tension points?")

    has_chunk_text = "chunk_text" in chunk_level_df.columns
    if not has_chunk_text:
        st.info("The meeting data needs conversation text before questions can be answered.")
    elif not api_key:
        st.info("Question answering is not configured for this environment. Ask the workspace owner to set the service key.")
    elif st.button("Ask Question", type="primary"):
        try:
            records = tuple(
                (str(row.meeting_id), int(row.chunk_idx), str(row.chunk_text))
                for row in chunk_level_df.itertuples()
            )
            vector_store = build_faiss_store(records, api_key, embedding_model_name)
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            docs = retriever.invoke(question)
            context = "\n\n".join(
                f"Meeting {doc.metadata.get('meeting_id')} segment {doc.metadata.get('segment')}:\n{doc.page_content}"
                for doc in docs
            )
            llm = make_chat_openai(answer_model_name, api_key)
            answer = llm.invoke([
                ("system", "Answer using only the provided meeting context. Cite meeting IDs and segment numbers."),
                ("user", f"Context:\n{context}\n\nQuestion: {question}"),
            ])
            st.write(answer.content)
            st.markdown("**Relevant Meeting Segments**")
            st.dataframe(pd.DataFrame([doc.metadata for doc in docs]), use_container_width=True)
        except Exception as exc:
            st.error(f"Could not answer the question: {exc}")

with assistant_session_tab:
    if render_assistant_tab is None:
        st.info("Assistant session extension is not available in this environment.")
    else:
        render_assistant_tab(selected_assistant_session_dir, selected_assistant_session)

with source_tab:
    st.subheader("Loaded Data")
    status_rows = []
    for label, name in [
        ("Meeting scores", "metrics"),
        ("Conversation segments", "chunks"),
        ("Important moments", "moments"),
        ("Topic insights", "topics"),
        ("Meeting summaries", "summaries"),
    ]:
        status_rows.append({"Data": label, "Status": "Ready" if dataframes[name] is not None else "Waiting", "Source": statuses[name]})
    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    with st.expander("Selected meeting conversation segments"):
        st.dataframe(chunks_meeting.reset_index(drop=True), use_container_width=True)
