"""Web browser tool with Google Custom Search + direct URL fetch."""

from __future__ import annotations

import os
import re
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


class WebBrowserTool:
    def __init__(
        self,
        *,
        search_api_key: str | None = None,
        search_engine_id: str | None = None,
        user_agent: str | None = None,
        max_results: int = 5,
    ) -> None:
        self.search_api_key = search_api_key or os.getenv("GOOGLE_SEARCH_API_KEY")
        self.search_engine_id = search_engine_id or os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        self.user_agent = user_agent or (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
        self.max_results = max_results

    def run(
        self,
        *,
        query: str = "",
        url: str = "",
        max_content_length: int = 5000,
        max_links: int = 5,
    ) -> dict[str, Any]:
        if query and url:
            raise ValueError("Provide either query or url, not both.")
        if not query and not url:
            raise ValueError("Provide query or url.")

        if query:
            return self.search_web(query)
        return self.visit_url(url, max_content_length=max_content_length, max_links=max_links)

    def search_web(self, query: str) -> dict[str, Any]:
        api_key = self.search_api_key
        engine_id = self.search_engine_id
        if not api_key or not engine_id:
            return {
                "error": "Search API key or engine ID not configured. "
                "Set GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID.",
                "results": [],
                "results_count": 0,
            }

        url = "https://www.googleapis.com/customsearch/v1"
        params: dict[str, str | int] = {
            "key": api_key,
            "cx": engine_id,
            "q": query,
            "num": self.max_results,
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            items = data.get("items", []) or []
            results = []
            for item in items:
                results.append(
                    {
                        "title": item.get("title"),
                        "url": item.get("link"),
                        "snippet": item.get("snippet"),
                        "source": item.get("displayLink"),
                    }
                )
            return {
                "query": query,
                "results_count": len(results),
                "results": results,
                "summary": f"Found {len(results)} results for '{query}'.",
                "search_engine": "Google Custom Search",
            }
        except Exception as exc:
            return {
                "query": query,
                "results_count": 0,
                "results": [],
                "error": f"Search failed: {exc}",
                "search_engine": "Google Custom Search",
            }

    def visit_url(self, url: str, *, max_content_length: int, max_links: int) -> dict[str, Any]:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return {"error": f"Invalid URL: {url}"}

        headers = {"User-Agent": self.user_agent}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "meta", "noscript"]):
            tag.extract()

        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r"\n+", "\n", text)
        text = re.sub(r" +", " ", text)

        links = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.startswith("/"):
                href = f"{parsed.scheme}://{parsed.netloc}{href}"
            if href.startswith(("http://", "https://")):
                links.append({"text": link.get_text(strip=True) or href, "url": href})
            if len(links) >= max_links:
                break

        images = []
        for img in soup.find_all("img", src=True):
            src = img["src"]
            if src.startswith("/"):
                src = f"{parsed.scheme}://{parsed.netloc}{src}"
            if src.startswith(("http://", "https://")):
                images.append(src)
            if len(images) >= 3:
                break

        truncated = len(text) > max_content_length
        if truncated:
            text = text[:max_content_length]

        return {
            "title": title,
            "content": text,
            "url": response.url,
            "links": links,
            "images": images,
            "content_type": response.headers.get("Content-Type", ""),
            "content_length": len(text),
            "truncated": truncated,
            "summary": f"Fetched {response.url} ({len(text)} chars).",
        }
