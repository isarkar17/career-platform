"""
Embedder — creates OpenAI embeddings and stores chunks in Supabase pgvector.
Run after scraping. Safe to call multiple times (uses upsert pattern).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from supabase import create_client

EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE  = 50

_oai  = None
_supa = None


def _get_oai():
    global _oai
    if _oai is None:
        _oai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return _oai


def _get_supa():
    global _supa
    if _supa is None:
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_KEY", "")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")
        _supa = create_client(url, key)
    return _supa


def embed_texts(texts):
    """Create embeddings for a list of strings. Returns list of vectors."""
    resp = _get_oai().embeddings.create(model=EMBED_MODEL, input=texts)
    return [r.embedding for r in resp.data]


def embed_and_store(chunks, batch_size=BATCH_SIZE):
    """
    Embed a list of chunk dicts and insert into Supabase lenny_chunks table.

    Each chunk dict must have:
        content, title, url, source_type, topics, chunk_idx

    Returns total number of chunks stored.
    """
    if not chunks:
        print("  No chunks to embed.")
        return 0

    supa   = _get_supa()
    stored = 0
    total  = len(chunks)

    for i in range(0, total, batch_size):
        batch      = chunks[i:i + batch_size]
        texts      = [c["content"] for c in batch]
        embeddings = embed_texts(texts)

        rows = [
            {
                "content":     c["content"],
                "embedding":   emb,
                "title":       c.get("title", ""),
                "url":         c.get("url", ""),
                "source_type": c.get("source_type", "article"),
                "topics":      c.get("topics", "general"),
                "chunk_idx":   c.get("chunk_idx", 0),
            }
            for c, emb in zip(batch, embeddings)
        ]

        supa.table("lenny_chunks").insert(rows).execute()
        stored += len(rows)

        done = min(i + batch_size, total)
        pct  = round(done / total * 100)
        print(f"  Embedded and stored {done}/{total} ({pct}%) chunks...", flush=True)

    return stored


def get_total_chunks():
    """Return count of all chunks currently in Supabase."""
    try:
        r = _get_supa().table("lenny_chunks").select("id", count="exact").execute()
        return r.count
    except Exception:
        return -1


def delete_chunks_for_url(url):
    """Remove all chunks for a given URL (e.g. to force re-scraping)."""
    _get_supa().table("lenny_chunks").delete().eq("url", url).execute()
    print(f"  Deleted all chunks for: {url}")
