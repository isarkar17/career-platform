"""
Retriever — queries Supabase pgvector using semantic search.
Shared by all three tools (Career Coach, Mock Interviewer, Scorecard).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from supabase import create_client
from config import OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY

oai  = OpenAI(api_key=OPENAI_API_KEY)

supa = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supa = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"[retriever] WARNING: Could not connect to Supabase: {e}", flush=True)


def _embed(text):
    """Embed a single text string using OpenAI text-embedding-3-small."""
    resp = oai.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    )
    return resp.data[0].embedding


def retrieve(query, n=5, topics=None):
    """
    Semantic search over Lenny's archive in Supabase.

    Args:
        query:  Natural language query string
        n:      Number of results to return (default 5)
        topics: Optional list of topic tags to filter by

    Returns:
        List of dicts with keys: text, title, url, type, relevance
    """
    if not supa:
        return []

    query_vec    = _embed(query)
    topic_filter = topics[0] if topics else None

    result = supa.rpc("match_lenny_chunks", {
        "query_embedding": query_vec,
        "match_threshold": 0.35,
        "match_count":     n,
        "filter_topics":   topic_filter
    }).execute()

    return [
        {
            "text":      r["content"],
            "title":     r.get("title", ""),
            "url":       r.get("url", ""),
            "type":      r.get("source_type", ""),
            "relevance": round(r.get("similarity", 0), 3)
        }
        for r in (result.data or [])
    ]


def build_context(chunks, max_chars=4000):
    """
    Format retrieved chunks as a readable context string for Claude.
    Stops adding chunks once max_chars is reached.
    """
    parts, total = [], 0
    for i, c in enumerate(chunks, 1):
        segment = f"[Source {i}: {c['title']}]\n{c['text']}"
        if total + len(segment) > max_chars:
            break
        parts.append(segment)
        total += len(segment)
    return "\n\n---\n\n".join(parts)


def get_chunk_count():
    """Return the total number of chunks stored in Supabase."""
    if not supa:
        return -1
    try:
        r = supa.table("lenny_chunks").select("id", count="exact").execute()
        return r.count
    except Exception:
        return -1
