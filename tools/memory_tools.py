from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from vectordbsupabase import SupabaseVectorDB

SHARED_AGENT_NAME = "shared_context"

vector_db = SupabaseVectorDB()


def store_user_context(
    user_id: str,
    agent: str,
    content: str,
    embedding: Optional[List[float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Store user context in shared Supabase vector memory (no agent separation)."""
    merged_metadata: Dict[str, Any] = {"source_agent": agent}
    if metadata:
        merged_metadata.update(metadata)
    try:
        vector = embedding or vector_db.embed_text(content)
        return vector_db.upsert_record(
            user_id=user_id,
            agent=SHARED_AGENT_NAME,
            content=content,
            embedding=vector,
            metadata=merged_metadata,
        )
    except Exception as exc:
        print(f"Error storing context: {exc}")
        return ""


def retrieve_user_context(
    user_id: str,
    agent: str,
    query_embedding: List[float],
    top_k: int = 5,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """Retrieve similar context entries for a user across all agents using vector search."""
    try:
        return vector_db.similarity_search(
            user_id=user_id,
            agent=None,
            query_embedding=query_embedding,
            match_count=top_k,
            match_threshold=min_score,
        )
    except Exception as exc:
        print(f"Error retrieving context: {exc}")
        return []


def supabase_vector_schema_sql() -> str:
    """Return SQL required to provision the Supabase vector table and RPC."""
    try:
        return vector_db.schema_sql()
    except Exception as exc:
        print(f"Error generating schema SQL: {exc}")
        return ""


# ---------------------------------------------------------------------------
# Tool-wrapped helpers (agent-specific) for recalling/storing short notes
# ---------------------------------------------------------------------------


def _render_rows(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No recent memory found."
    return "\n".join(
        f"- [{row.get('metadata', {}).get('source_agent', 'unknown')}] "
        f"{row.get('metadata', {}).get('user_message', '')} -> "
        f"{row.get('metadata', {}).get('agent_response', '')}"
        for row in rows
    )


@tool
def search_user_memory(query: str, user_id: str) -> str:
    """Search recent Supabase memory for this user (shared across agents)."""
    try:
        emb = vector_db.embed_text(query)
        rows = retrieve_user_context(
            user_id=str(user_id),
            agent=SHARED_AGENT_NAME,
            query_embedding=emb,
            top_k=5,
            min_score=0.25,
        )
    except Exception as exc:
        return f"Memory search unavailable: {exc}"
    return _render_rows(rows)


@tool
def store_user_note(note: str, user_id: str) -> str:
    """Store a short note to shared Supabase memory."""
    try:
        store_user_context(
            user_id=str(user_id),
            agent="general_agent",
            content=note,
            metadata={"user_message": note, "agent_response": "stored_note"},
        )
        return "Saved to memory."
    except Exception as exc:
        return f"Could not save memory: {exc}"

