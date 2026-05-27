"""
Manifest system — tracks which URLs have been scraped.
Enables incremental updates: only new content is processed on each refresh run.
"""

import json
import os
from datetime import datetime, timezone

MANIFEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.json")


def load():
    """Load manifest from disk. Returns empty manifest if not found."""
    if not os.path.exists(MANIFEST_PATH):
        return {"scraped": {}, "last_run": None, "version": "1.0"}
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def save(manifest):
    """Save manifest to disk with updated timestamp."""
    manifest["last_run"] = datetime.now(timezone.utc).isoformat()
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def mark_done(manifest, url, chunk_count, status="ok"):
    """Record that a URL has been processed."""
    manifest["scraped"][url] = {
        "scraped_at":  datetime.now(timezone.utc).isoformat(),
        "chunk_count": chunk_count,
        "status":      status
    }


def get_new_urls(manifest, all_urls):
    """Return URLs that have not yet been successfully scraped."""
    done = {
        url for url, data in manifest["scraped"].items()
        if data.get("status") == "ok"
    }
    return [u for u in all_urls if u not in done]


def summary(manifest):
    """Return a human-readable summary of manifest state."""
    total   = len(manifest["scraped"])
    ok      = sum(1 for v in manifest["scraped"].values() if v.get("status") == "ok")
    skipped = total - ok
    return (
        f"{ok} URLs scraped successfully, "
        f"{skipped} skipped (paywalled or error). "
        f"Last run: {manifest.get('last_run', 'never')}"
    )


def reset(url, manifest):
    """Force re-scrape of a specific URL by removing it from manifest."""
    manifest["scraped"].pop(url, None)
