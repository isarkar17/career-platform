"""
run_full.py — First-time pipeline run.

Run this ONCE to scrape all known URLs and populate Supabase.
After this, use refresh.py for incremental updates.

Usage (from artifacts/flask-app/ directory):
    python3 pipeline/run_full.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.manifest import load, save, mark_done, get_new_urls, summary
from pipeline.scraper   import scrape_new
from pipeline.embedder  import embed_and_store, get_total_chunks
from pipeline.all_urls  import ALL_ARTICLE_URLS, ALL_PODCAST_URLS

print("=" * 60)
print("Career Platform — Full Content Pipeline")
print("=" * 60)

manifest = load()
print(f"\nManifest status: {summary(manifest)}")

new_articles = get_new_urls(manifest, ALL_ARTICLE_URLS)
new_podcasts = [(u, t) for u, t in ALL_PODCAST_URLS
                if get_new_urls(manifest, [u])]

print(f"\nArticles to scrape:  {len(new_articles)}")
print(f"Podcasts to scrape:  {len(new_podcasts)}")
print(f"Already done:        {len(ALL_ARTICLE_URLS) - len(new_articles)} articles, "
      f"{len(ALL_PODCAST_URLS) - len(new_podcasts)} podcasts\n")

if not new_articles and not new_podcasts:
    print("Nothing new to scrape — Supabase is already up to date.")
    print(f"Total chunks in Supabase: {get_total_chunks()}")
    sys.exit(0)

print("Scraping content...")
chunks = scrape_new(new_articles, new_podcasts)

if not chunks:
    print("\nNo chunks produced — all URLs were paywalled or errored.")
    sys.exit(1)

print(f"\nEmbedding {len(chunks)} chunks into Supabase...")
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

print(f"\n{'=' * 60}")
print(f"Done! {stored} chunks stored in Supabase.")
print(f"Total chunks in Supabase: {get_total_chunks()}")
print(f"Updated manifest: {summary(manifest)}")
