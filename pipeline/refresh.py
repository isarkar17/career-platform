"""
refresh.py — Incremental content refresh.

Only processes URLs not already in the manifest.
Safe to run manually at any time.

Usage (from artifacts/flask-app/ directory):
    python3 pipeline/refresh.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.manifest import load, save, mark_done, get_new_urls, summary
from pipeline.scraper   import scrape_new
from pipeline.embedder  import embed_and_store, get_total_chunks
from pipeline.all_urls  import ALL_ARTICLE_URLS, ALL_PODCAST_URLS

print("=== Content refresh ===")

manifest = load()
print(f"Manifest: {summary(manifest)}")

new_articles = get_new_urls(manifest, ALL_ARTICLE_URLS)
new_podcasts = [(u, t) for u, t in ALL_PODCAST_URLS
                if get_new_urls(manifest, [u])]

if not new_articles and not new_podcasts:
    print("Nothing new — Supabase is fully up to date.")
    print(f"Total chunks: {get_total_chunks()}")
    sys.exit(0)

print(f"New: {len(new_articles)} articles, {len(new_podcasts)} podcasts")

chunks = scrape_new(new_articles, new_podcasts)

if not chunks:
    print("No chunks produced (all new URLs paywalled or errored).")
    for url in new_articles:
        mark_done(manifest, url, 0, status="paywalled")
    for url, _ in new_podcasts:
        mark_done(manifest, url, 0, status="paywalled")
    save(manifest)
    sys.exit(0)

stored = embed_and_store(chunks)

url_to_chunks = {}
for c in chunks:
    url_to_chunks.setdefault(c["url"], 0)
    url_to_chunks[c["url"]] += 1

for url in new_articles:
    count = url_to_chunks.get(url, 0)
    mark_done(manifest, url, count, status="ok" if count else "paywalled")

for url, _ in new_podcasts:
    count = url_to_chunks.get(url, 0)
    mark_done(manifest, url, count, status="ok" if count else "paywalled")

save(manifest)

print(f"Refresh complete. {stored} new chunks added.")
print(f"Total chunks in Supabase: {get_total_chunks()}")
