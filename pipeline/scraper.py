"""
Content scraper — fetches Lenny's free articles and podcast transcripts.
Skips paywalled content (too short) and respects the server with delays.
"""

import time
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; career-platform-research-bot/1.0; "
        "+https://github.com/your-username/career-platform)"
    )
}
REQUEST_DELAY = 2.5   # seconds between requests — be respectful
MIN_CONTENT   = 600   # characters — shorter = paywalled

TOPIC_MAP = {
    "job-interview":    ["interviews", "hiring", "careers"],
    "interview":        ["interviews", "hiring", "careers"],
    "job-market":       ["job-market", "skills", "careers"],
    "retention":        ["general", "careers"],
    "consumer-apps":    ["careers", "general"],
    "b2b":              ["general"],
    "behavioral":       ["general", "careers"],
    "ai-tools":         ["skills", "job-market"],
    "community":        ["general"],
    "product-market":   ["general"],
    "growth":           ["general"],
}


def _infer_topics(url):
    slug = url.split("/p/")[-1].lower()
    for key, topics in TOPIC_MAP.items():
        if key in slug:
            return topics
    return ["general"]


def scrape_url(url, source_type="article", topics=None):
    """
    Fetch a single URL and extract its main content.
    Returns a document dict or None if unavailable / paywalled.
    """
    try:
        r = requests.get(url, headers=HEADERS, timeout=8)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"    Network error {url}: {e}")
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    title_el = soup.find("h1")
    title    = title_el.get_text(strip=True) if title_el else url.split("/")[-1]

    body = soup.find("div", class_="body markup")
    if not body:
        return None

    text = body.get_text(separator="\n", strip=True)
    if len(text) < MIN_CONTENT:
        return None  # Paywalled — content truncated

    return {
        "url":    url,
        "title":  title,
        "text":   text,
        "type":   source_type,
        "topics": topics or _infer_topics(url),
    }


def chunk_text(text, max_chars=1800, overlap=200):
    """Split text into overlapping character-bounded chunks."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # Prefer splitting at paragraph boundary
        if end < len(text):
            last_nl = text.rfind("\n", start, end)
            if last_nl > start + 600:
                end = last_nl
        chunk = text[start:end].strip()
        if len(chunk) > 120:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def scrape_new(new_article_urls, new_podcast_url_pairs):
    """
    Scrape a list of new article URLs and podcast (url, topics) pairs.
    Returns a list of chunk dicts ready for embedding.
    """
    all_chunks = []

    for url in new_article_urls:
        doc = scrape_url(url, source_type="article")
        if doc:
            parts = chunk_text(doc["text"])
            for i, part in enumerate(parts):
                all_chunks.append({
                    "url":         url,
                    "title":       doc["title"],
                    "content":     part,
                    "source_type": "article",
                    "topics":      ",".join(doc["topics"]),
                    "chunk_idx":   i,
                })
            print(f"  article  ({len(parts):3d} chunks): {doc['title'][:55]}")
        else:
            print(f"  skipped  (paywalled or error): {url.split('/')[-1]}")
        time.sleep(REQUEST_DELAY)

    for url, topics in new_podcast_url_pairs:
        doc = scrape_url(url, source_type="podcast", topics=topics)
        if doc:
            parts = chunk_text(doc["text"])
            for i, part in enumerate(parts):
                all_chunks.append({
                    "url":         url,
                    "title":       doc["title"],
                    "content":     part,
                    "source_type": "podcast",
                    "topics":      ",".join(topics),
                    "chunk_idx":   i,
                })
            print(f"  podcast  ({len(parts):3d} chunks): {doc['title'][:55]}")
        else:
            print(f"  skipped  (paywalled or error): {url.split('/')[-1]}")
        time.sleep(REQUEST_DELAY)

    return all_chunks
