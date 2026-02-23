"""Shared agent infrastructure — model, vector DB, memory, and run loop.

Every agent imports from here instead of duplicating setup boilerplate.
This saves ~80 lines per agent and ensures a single model / vector-DB
instance is shared across all agents (lower memory, faster cold-start).
"""

import logging
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

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
    except Exception:
        rows = []

    if not rows:
        try:
            rows = vector_db.latest_records(user_id=str(user_id), agent=None, limit=5)
        except Exception:
            rows = []

    if not rows:
        return None

    return "Recent conversation history:\n" + "".join(
        f"User: {r.get('metadata', {}).get('user_message')}\n"
        f"Agent: {r.get('metadata', {}).get('agent_response')}\n"
        for r in rows
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
) -> str:
    """Invoke a pre-built LangChain agent with memory context and persist the turn.

    1. Looks up relevant memory → optional system message.
    2. Invokes the agent.
    3. Extracts the reply text.
    4. Persists the conversation turn to shared Supabase memory.
    """
    print(f"\n{'='*60}")
    print(f"USER {user_id}: {user_message}")
    print(f"{'='*60}\n")

    # Build messages — skip system message if no memory found
    messages = []
    sys_content = build_system_message(user_id, user_message)
    if sys_content:
        messages.append({"role": "system", "content": sys_content})
    messages.append({"role": "user", "content": user_message})

    result = agent.invoke(
        {"messages": messages},
        config={"user_id": user_id},
    )

    final_message = result["messages"][-1]
    agent_reply = normalize_content(final_message.content)
    print("AGENT:", final_message.content)

    # Persist conversation turn to shared Supabase memory
    try:
        store_user_context(
            user_id=str(user_id),
            agent=agent_name,
            content=f"User: {user_message}\nAgent: {agent_reply}",
            metadata={"user_message": user_message, "agent_response": agent_reply},
        )
    except Exception as exc:
        logger.warning(f"Error persisting {agent_name} memory: {exc}")

    return agent_reply
