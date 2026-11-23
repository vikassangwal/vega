"""
JARVIS OMEGA - Web Tools
Built-in tools for web search, fetch, and analysis.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import re
import urllib.parse
from dataclasses import dataclass
from typing import Any, Optional

import aiohttp
from bs4 import BeautifulSoup

from tools.tool_registry import ToolRegistry, ToolMetadata, ToolParameter
from config import get_settings, settings

logger = logging.getLogger("vega.tools.web")


# =============================================================================
# SEARCH PROVIDERS
# =============================================================================

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str


class SearchProvider:
    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        raise NotImplementedError


class DuckDuckGoSearch(SearchProvider):
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        try:
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            headers = {"User-Agent": "Mozilla/5.0 (compatible; VEGA/1.0)"}

            async with self.session.get(url, headers=headers) as resp:
                html = await resp.text()

            soup = BeautifulSoup(html, "html.parser")
            results = []

            for result in soup.select(".result__body")[:max_results]:
                title_el = result.select_one(".result__title")
                snippet_el = result.select_one(".result__snippet")
                url_el = result.select_one(".result__url")

                if title_el:
                    title = title_el.get_text(strip=True)
                    snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                    url = url_el.get_text(strip=True) if url_el else ""

                    results.append(SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source="duckduckgo",
                    ))

            return results
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []


class BraveSearch(SearchProvider):
    def __init__(self, session: aiohttp.ClientSession, api_key: str):
        self.session = session
        self.api_key = api_key

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        if not self.api_key:
            return []

        try:
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {
                "X-Subscription-Token": self.api_key,
                "Accept": "application/json",
            }
            params = {"q": query, "count": max_results}

            async with self.session.get(url, headers=headers, params=params) as resp:
                data = await resp.json()

            results = []
            for item in data.get("web", {}).get("results", [])[:max_results]:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("description", ""),
                    source="brave",
                ))

            return results
        except Exception as e:
            logger.error(f"Brave search error: {e}")
            return []


# =============================================================================
# WEB TOOLS IMPLEMENTATION
# =============================================================================

class WebTools:
    def __init__(self):
        self._session: aiohttp.ClientSession | None = None
        self._search_providers: list = []
        self._cache: dict[str, Any] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"User-Agent": "VEGA/1.0 (compatible; Research Bot)"},
            )

            self._search_providers = [
                DuckDuckGoSearch(self._session),
            ]

            brave_key = get_settings().get("brave_api_key") if hasattr(get_settings(), 'brave_api_key') else None
            if brave_key:
                self._search_providers.append(BraveSearch(self._session, brave_key))

        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def web_search(self, params: dict) -> dict:
        query = params.get("query", "")
        max_results = params.get("max_results", 10)
        sources = params.get("sources", ["duckduckgo"])

        if not query:
            return {"success": False, "error": "Query required"}

        cache_key = hashlib.md5(f"{query}:{max_results}".encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        await self._get_session()

        all_results = []
        for provider in self._search_providers:
            provider_name = provider.__class__.__name__.replace("Search", "").lower()
            if provider_name in sources or "all" in sources:
                results = await provider.search(query, max_results)
                all_results.extend(results)

        all_results.sort(key=lambda r: r.relevance_score, reverse=True)

        result = {
            "success": True,
            "query": query,
            "results": [
                {"title": r.title, "url": r.url, "snippet": r.snippet, "source": r.source}
                for r in all_results[:max_results]
            ],
            "total": len(all_results),
        }

        self._cache[cache_key] = result
        return result

    async def fetch_page(self, params: dict) -> dict:
        url = params.get("url", "")
        extract = params.get("extract", True)
        selectors = params.get("selectors", [])
        timeout = params.get("timeout", 30)

        if not url:
            return {"success": False, "error": "URL required"}

        if not url.startswith(("http://", "https://")):
            return {"success": False, "error": "Invalid URL (must be http/https)"}

        try:
            session = await self._get_session()

            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status >= 400:
                    return {"success": False, "error": f"HTTP {resp.status}"}

                content_type = resp.headers.get("Content-Type", "")
                html = await resp.text()

                result = {
                    "success": True,
                    "url": url,
                    "status": resp.status,
                    "content_type": content_type,
                }

                if extract or selectors:
                    soup = BeautifulSoup(html, "html.parser")

                    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                        tag.decompose()

                    if selectors:
                        extracted = {}
                        for sel in selectors:
                            elements = soup.select(sel)
                            extracted[sel] = [el.get_text(strip=True) for el in elements]
                        result["extracted"] = extracted
                    else:
                        main_content = soup.get_text(separator="\n", strip=True)
                        lines = [line.strip() for line in main_content.split("\n") if line.strip()]
                        result["content"] = "\n".join(lines[:2000])

                        title_tag = soup.find("title")
                        if title_tag:
                            result["title"] = title_tag.get_text(strip=True)

                        meta_desc = soup.find("meta", attrs={"name": "description"})
                        if meta_desc:
                            result["description"] = meta_desc.get("content", "")

                return result

        except asyncio.TimeoutError:
            return {"success": False, "error": "Request timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def download_file(self, params: dict) -> dict:
        url = params.get("url", "")
        output = params.get("output", "")
        max_size = params.get("max_size", 100 * 1024 * 1024)

        if not url or not output:
            return {"success": False, "error": "URL and output path required"}

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            session = await self._get_session()

            async with session.get(url, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                if resp.status >= 400:
                    return {"success": False, "error": f"HTTP {resp.status}"}

                content_length = resp.headers.get("Content-Length")
                if content_length and int(content_length) > max_size:
                    return {"success": False, "error": "File too large"}

                downloaded = 0
                with open(output_path, "wb") as f:
                    async for chunk in resp.content.iter_chunked(8192):
                        downloaded += len(chunk)
                        if downloaded > max_size:
                            return {"success": False, "error": "File too large"}
                        f.write(chunk)

            return {"success": True, "path": str(output_path), "size": downloaded}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def fetch_rss(self, params: dict) -> dict:
        import feedparser

        url = params.get("url", "")
        max_items = params.get("max_items", 20)

        if not url:
            return {"success": False, "error": "Feed URL required"}

        try:
            session = await self._get_session()

            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                content = await resp.text()

            feed = feedparser.parse(content)

            items = []
            for entry in feed.entries[:max_items]:
                items.append({
                    "title": entry.get("title", ""),
                    "link": entry.get("link", ""),
                    "summary": entry.get("summary", ""),
                    "published": entry.get("published", ""),
                    "author": entry.get("author", ""),
                })

            return {
                "success": True,
                "feed_title": feed.feed.get("title", ""),
                "feed_link": feed.feed.get("link", ""),
                "items": items,
                "total": len(items),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}


_web_tools = WebTools()


async def web_search(params: dict) -> dict:
    return await _web_tools.web_search(params)


async def fetch_page(params: dict) -> dict:
    return await _web_tools.fetch_page(params)


async def download_file(params: dict) -> dict:
    return await _web_tools.download_file(params)


async def fetch_rss(params: dict) -> dict:
    return await _web_tools.fetch_rss(params)


async def extract_content(params: dict) -> dict:
    url = params.get("url", "")
    selectors = params.get("selectors", {})
    attributes = params.get("attributes", {})

    if not url:
        return {"success": False, "error": "URL required"}

    try:
        session = await _web_tools._get_session()

        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            html = await resp.text()

        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        result = {"success": True, "url": url, "data": {}}

        for field, selector in selectors.items():
            elements = soup.select(selector)
            attr = attributes.get(field, "text")

            if attr == "text":
                values = [el.get_text(strip=True) for el in elements]
            elif attr == "html":
                values = [str(el) for el in elements]
            else:
                values = [el.get(attr, "") for el in elements]

            result["data"][field] = values[0] if len(values) == 1 else values

        return result

    except Exception as e:
        return {"success": False, "error": str(e)}


async def take_screenshot(params: dict) -> dict:
    try:
        from playwright.async_api import async_playwright

        url = params.get("url", "")
        output = params.get("output", "screenshot.png")
        full_page = params.get("full_page", True)
        viewport = params.get("viewport", {"width": 1280, "height": 720})

        if not url:
            return {"success": False, "error": "URL required"}

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(viewport=viewport)
            await page.goto(url, wait_until="networkidle")
            await page.screenshot(path=str(output_path), full_page=full_page)
            await browser.close()

        return {
            "success": True,
            "path": str(output_path),
            "size": Path(output_path).stat().st_size,
        }

    except ImportError:
        return {"success": False, "error": "Playwright not installed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def register_web_tools(registry: ToolRegistry) -> None:
    tools = [
        ("web_search", web_search, "Search the web", [
            ToolParameter("query", "string", "Search query", required=True),
            ToolParameter("max_results", "integer", "Maximum results", default=10),
            ToolParameter("sources", "array", "Search sources: duckduckgo, brave", default=["duckduckgo"]),
        ], {"requires_approval": False, "sandboxed": True}),

        ("fetch_page", fetch_page, "Fetch and extract page content", [
            ToolParameter("url", "string", "URL to fetch", required=True),
            ToolParameter("extract", "boolean", "Extract main content", default=True),
            ToolParameter("selectors", "array", "CSS selectors for specific extraction"),
            ToolParameter("timeout", "integer", "Request timeout (seconds)", default=30),
        ], {"requires_approval": False, "sandboxed": True}),

        ("download_file", download_file, "Download file from URL", [
            ToolParameter("url", "string", "File URL", required=True),
            ToolParameter("output", "string", "Output path", required=True),
            ToolParameter("max_size", "integer", "Max file size (bytes)", default=104857600),
        ], {"requires_approval": True, "sandboxed": True}),

        ("fetch_rss", fetch_rss, "Fetch and parse RSS/Atom feed", [
            ToolParameter("url", "string", "Feed URL", required=True),
            ToolParameter("max_items", "integer", "Maximum items", default=20),
        ], {"requires_approval": False, "sandboxed": True}),

        ("extract_content", extract_content, "Extract structured content with selectors", [
            ToolParameter("url", "string", "Page URL", required=True),
            ToolParameter("selectors", "object", "Field -> CSS selector mapping", required=True),
            ToolParameter("attributes", "object", "Field -> attribute to extract (text, html, href, etc.)"),
        ], {"requires_approval": False, "sandboxed": True}),

        ("take_screenshot", take_screenshot, "Take webpage screenshot", [
            ToolParameter("url", "string", "Page URL", required=True),
            ToolParameter("output", "string", "Output path", default="screenshot.png"),
            ToolParameter("full_page", "boolean", "Full page screenshot", default=True),
            ToolParameter("viewport", "object", "Viewport size", default={"width": 1280, "height": 720}),
        ], {"requires_approval": False, "sandboxed": True}),
    ]

    for name, func, desc, params, extra in tools:
        metadata = ToolMetadata(
            name=name,
            description=desc,
            category="web",
            parameters=params,
            returns={"type": "object"},
            sandboxed=extra.get("sandboxed", True),
            requires_approval=extra.get("requires_approval", False),
            tags=["web", "internet", "search"],
        )
        registry.register(name, func, metadata)

    logger.info("Registered web tools")


__all__ = [
    "register_web_tools",
    "WebTools",
    "SearchResult",
]