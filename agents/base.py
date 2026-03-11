"""Shared agent infrastructure — model, vector DB, memory, and run loop.

Every agent imports from here instead of duplicating setup boilerplate.
This saves ~80 lines per agent and ensures a single model / vector-DB
instance is shared across all agents (lower memory, faster cold-start).
"""

import logging
import os
import time

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver

from tools.memory_tools import retrieve_user_context, store_user_context
from vectordbsupabase import SupabaseVectorDB

load_dotenv()

logger = logging.getLogger(__name__)

# ── Shared singletons (created once, reused by all agents) ────────────────

_google_api_key = os.getenv("GOOGLE_API_KEY")
if not _google_api_key:
    raise RuntimeError("Missing GOOGLE_API_KEY environment variable")
os.environ["GOOGLE_API_KEY"] = _google_api_key

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
vector_db = SupabaseVectorDB()
checkpointer = MemorySaver()

# Track which threads have already had at least one turn (to avoid
# injecting stale vector-memory context when the checkpointer already
# replays the in-session history).
# Uses a dict with timestamps so stale entries can be evicted, preventing
# unbounded memory growth.
_active_threads: dict[str, float] = {}   # thread_id → first-seen timestamp
_THREAD_TTL: float = 3600.0              # evict after 1 hour


# ── Shared utilities ──────────────────────────────────────────────────────

def build_system_message(user_id: int, user_message: str) -> str | None:
    """Fetch recent context from Supabase vector memory for a user.

    Returns a formatted history string or *None* if nothing relevant was found.
    """
    try:
        query_emb = vector_db.embed_text(user_message)
        rows = retrieve_user_context(
            user_id=str(user_id),
            agent="shared_context",
            query_embedding=query_emb,
            top_k=5,
            min_score=0.3,
        )
    except Exception as exc:
        logger.warning("Vector memory similarity search failed: %s", exc)
        rows = []

    if not rows:
        try:
            rows = vector_db.latest_records(user_id=str(user_id), agent=None, limit=5)
        except Exception as exc:
            logger.warning("Vector memory latest-records fallback failed: %s", exc)
            rows = []

    if not rows:
        return None

    parts: list[str] = []
    for r in rows:
        meta = r.get("metadata", {})
        user_msg = meta.get("user_message", "")
        agent_msg = meta.get("agent_response", "")
        source = meta.get("source_agent", "unknown")
        if user_msg or agent_msg:
            parts.append(f"[{source}] User: {user_msg}\nAgent: {agent_msg}")

    if not parts:
        return None

    return (
        "Relevant prior context from long-term memory (use if helpful, "
        "ignore if stale or irrelevant):\n" + "\n---\n".join(parts)
    )


def normalize_content(content) -> str:
    """Extract plain text from a LangChain message content field.

    Handles both plain-string content and the list-of-parts format
    returned by some Gemini models.
    """
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content) if content else ""


def run_agent_turn(
    agent_name: str,
    agent,
    user_message: str,
    user_id: int,
    session_id: str | None = None,
) -> str:
    """Invoke a pre-built LangChain agent with memory context and persist the turn.

    1. On the **first turn** of a session, looks up relevant long-term memory
       from Supabase and injects it as a system message so the agent has cross-
       session continuity. On subsequent turns the checkpointer already replays
       prior messages, so duplicate injection is skipped.
    2. Invokes the agent with a checkpointed thread_id so LangGraph
       replays prior turns automatically within the same session.
    3. Extracts the reply text.
    4. Persists the conversation turn to shared Supabase memory.
    """
    start = time.perf_counter()
    logger.info("┌── %s | user=%s | session=%s", agent_name, user_id, session_id or "default")
    logger.info("│  Q: %.200s", user_message)

    # Use thread_id for LangGraph checkpointing (preserves in-session context)
    thread_id = f"{agent_name}_{user_id}_{session_id or 'default'}"

    # Build messages — only inject long-term memory on the first turn of this
    # thread to avoid duplicating context that the checkpointer already replays.
    messages = []
    now = time.perf_counter()
    is_first_turn = thread_id not in _active_threads or (now - _active_threads[thread_id]) > _THREAD_TTL
    if is_first_turn:
        sys_content = build_system_message(user_id, user_message)
        if sys_content:
            messages.append({"role": "system", "content": sys_content})
        _active_threads[thread_id] = now
        # Evict stale entries to prevent unbounded growth
        if len(_active_threads) > 200:
            cutoff = now - _THREAD_TTL
            stale = [k for k, ts in _active_threads.items() if ts < cutoff]
            for k in stale:
                del _active_threads[k]

    messages.append({"role": "user", "content": user_message})

    result = agent.invoke(
        {"messages": messages},
        config={"configurable": {"thread_id": thread_id}},
    )

    # Extract the reply — prefer the last message, but fall back to the last
    # AI message with non-empty text (handles cases where the model returns an
    # empty final turn after a tool call).
    final_message = result["messages"][-1]
    agent_reply = normalize_content(final_message.content)

    if not agent_reply.strip():
        # Walk backwards through messages to find the last AI reply with text
        for msg in reversed(result["messages"]):
            role = getattr(msg, "type", None) or getattr(msg, "role", "")
            if role in ("ai", "assistant"):
                candidate = normalize_content(msg.content)
                if candidate.strip():
                    agent_reply = candidate
                    logger.info("│  (fell back to earlier AI message)")
                    break
        if not agent_reply.strip():
            agent_reply = "I processed your request but wasn't able to produce a response. Please try rephrasing your question."
            logger.warning("│  Agent produced empty reply for: %s", user_message)

    elapsed = time.perf_counter() - start
    logger.info("│  A: %.300s", agent_reply)
    logger.info("└── %s done in %.2fs", agent_name, elapsed)

    # Persist conversation turn to shared Supabase memory
    try:
        store_user_context(
            user_id=str(user_id),
            agent=agent_name,
            content=f"User: {user_message}\nAgent: {agent_reply}",
            metadata={"user_message": user_message, "agent_response": agent_reply},
        )
    except Exception as exc:
        logger.warning("Error persisting %s memory: %s", agent_name, exc)

    return agent_reply
