from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

DEFAULT_TABLE = os.getenv("SUPABASE_VECTOR_TABLE", "agent_memory")
DEFAULT_RPC_FN = os.getenv("SUPABASE_VECTOR_MATCH_FN", "match_agent_context")
DEFAULT_EMBED_DIM = int(os.getenv("SUPABASE_VECTOR_DIM", "768"))
DEFAULT_EMBED_MODEL = os.getenv("SUPABASE_EMBEDDING_MODEL", "models/embedding-001")


def build_schema_sql(
    table_name: str = DEFAULT_TABLE,
    embedding_dim: int = DEFAULT_EMBED_DIM,
    rpc_fn: str = DEFAULT_RPC_FN,
) -> str:
    return f"""
create extension if not exists vector;
create extension if not exists pgcrypto;

create table if not exists {table_name} (
    id uuid primary key default gen_random_uuid(),
    user_id text not null,
    agent text not null,
    content text not null,
    metadata jsonb default '{{}}'::jsonb,
    embedding vector({embedding_dim}) not null,
    created_at timestamptz not null default now()
);

create index if not exists {table_name}_user_agent_idx on {table_name} (user_id, agent);
create index if not exists {table_name}_ivfflat_embedding_idx on {table_name} using ivfflat (embedding vector_cosine_ops) with (lists = 100);

create or replace function {rpc_fn}(
    query_embedding vector({embedding_dim}),
    match_count int default 5,
    match_threshold float default 0.0,
    filter_user_id text default null,
    filter_agent text default null
) returns table (
    id uuid,
    user_id text,
    agent text,
    content text,
    metadata jsonb,
    created_at timestamptz,
    similarity float
) language plpgsql as $$
begin
    return query
    select
        m.id,
        m.user_id,
        m.agent,
        m.content,
        m.metadata,
        m.created_at,
        1 - (m.embedding <=> query_embedding) as similarity
    from {table_name} as m
    where (filter_user_id is null or m.user_id = filter_user_id)
      and (filter_agent is null or m.agent = filter_agent)
      and (match_threshold <= 0 or 1 - (m.embedding <=> query_embedding) >= match_threshold)
    order by m.embedding <=> query_embedding
    limit match_count;
end;
$$;
"""


def get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_API_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_API_KEY must be set in the environment.")
    return create_client(url, key)


class SupabaseVectorDB:
    _genai_module = None  # Class-level cache: import + configure genai once

    def __init__(
        self,
        client: Optional[Client] = None,
        table_name: str = DEFAULT_TABLE,
        rpc_fn: str = DEFAULT_RPC_FN,
        embedding_dim: int = DEFAULT_EMBED_DIM,
    ) -> None:
        self.client = client or get_supabase_client()
        self.table_name = table_name
        self.rpc_fn = rpc_fn
        self.embedding_dim = embedding_dim

    def schema_sql(self) -> str:
        return build_schema_sql(self.table_name, self.embedding_dim, self.rpc_fn)

    @classmethod
    def _init_genai(cls):
        """Import and configure google.generativeai exactly once."""
        if cls._genai_module is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("GOOGLE_API_KEY is required for automatic embeddings.")
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            cls._genai_module = genai
        return cls._genai_module

    def embed_text(self, text: str, model: Optional[str] = None) -> List[float]:
        genai = self._init_genai()
        model_name = model or DEFAULT_EMBED_MODEL
        response = genai.embed_content(model=model_name, content=text)
        embedding = response.get("embedding") if isinstance(response, dict) else None
        if not embedding:
            raise RuntimeError("Embedding service returned no vector.")
        return self._validate_embedding(embedding)

    def upsert_record(
        self,
        user_id: str,
        agent: str,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        vector = self._validate_embedding(embedding)
        payload: Dict[str, Any] = {
            "user_id": user_id,
            "agent": agent,
            "content": content,
            "metadata": metadata or {},
            "embedding": vector,
        }
        try:
            result = self.client.table(self.table_name).upsert(payload, returning="representation").execute()
        except Exception as exc:
            raise RuntimeError(f"Failed to upsert context: {exc}") from exc
        rows = getattr(result, "data", None)
        if rows and isinstance(rows, list) and rows[0].get("id"):
            return str(rows[0]["id"])
        return ""

    def similarity_search(
        self,
        user_id: Optional[str],
        agent: Optional[str],
        query_embedding: List[float],
        match_count: int = 5,
        match_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        vector = self._validate_embedding(query_embedding)
        params: Dict[str, Any] = {
            "query_embedding": vector,
            "match_count": match_count,
            "match_threshold": match_threshold,
            "filter_user_id": user_id,
            "filter_agent": agent,
        }
        try:
            result = self.client.rpc(self.rpc_fn, params).execute()
        except Exception as exc:
            raise RuntimeError(f"Failed to run similarity search: {exc}") from exc
        rows = getattr(result, "data", None)
        return rows if isinstance(rows, list) else []

    def latest_records(
        self,
        user_id: Optional[str],
        agent: Optional[str],
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        try:
            query = self.client.table(self.table_name).select("*").order("created_at", desc=True).limit(limit)
            if user_id:
                query = query.eq("user_id", user_id)
            if agent:
                query = query.eq("agent", agent)
            result = query.execute()
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch recent context: {exc}") from exc
        rows = getattr(result, "data", None)
        return rows if isinstance(rows, list) else []

    def _validate_embedding(self, embedding: List[float]) -> List[float]:
        if not isinstance(embedding, list):
            raise ValueError("Embedding must be a list of floats.")
        if len(embedding) != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.embedding_dim}, got {len(embedding)}.")
        return embedding
