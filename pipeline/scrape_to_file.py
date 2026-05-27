"""
Standalone scraper — no secrets needed. Scrapes all URLs and saves
chunks as JSON to /tmp/lenny_chunks.json for the embed step.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pipeline.scraper as _s
_s.REQUEST_DELAY = 0.8

from pipeline.scraper  import scrape_new
from pipeline.all_urls import ALL_ARTICLE_URLS, ALL_PODCAST_URLS

OUT = "/tmp/lenny_chunks.json"

print(f"Scraping {len(ALL_ARTICLE_URLS)} articles + {len(ALL_PODCAST_URLS)} podcasts...")
chunks = scrape_new(ALL_ARTICLE_URLS, ALL_PODCAST_URLS)

if not chunks:
    print("No chunks produced — all URLs may be paywalled.")
    sys.exit(1)

with open(OUT, "w") as f:
    json.dump(chunks, f)

print(f"Saved {len(chunks)} chunks to {OUT}")
