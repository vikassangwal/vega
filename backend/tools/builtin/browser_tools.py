"""
JARVIS OMEGA - Browser Tools
Built-in tools for browser automation (Playwright).
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from tools.tool_registry import ToolRegistry, ToolMetadata, ToolParameter

logger = logging.getLogger("vega.tools.browser")


# =============================================================================
# BROWSER MANAGER
# =============================================================================

@dataclass
class BrowserConfig:
    headless: bool = True
    viewport: dict = field(default_factory=lambda: {"width": 1280, "height": 720})
    user_agent: str | None = None
    proxy: str | None = None
    downloads_path: str | None = None
    stealth: bool = False  # Anti-detection
    timeout: int = 30000  # ms


class BrowserManager:
    """Manages Playwright browser instances."""

    def __init__(self):
        self._playwright = None
        self._browser = None
        self._contexts: dict[str, Any] = {}
        self._pages: dict[str, Any] = {}
        self._initialized = False

    async def initialize(self, config: BrowserConfig | None = None) -> bool:
        if self._initialized:
            return True

        try:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
            self._config = config or BrowserConfig()

            launch_options = {
                "headless": self._config.headless,
                "args": [
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-accelerated-2d-canvas",
                    "--no-first-run",
                    "--no-zygote",
                    "--disable-gpu",
                ],
            }

            if self._config.stealth:
                launch_options["args"].extend([
                    "--disable-blink-features=AutomationControlled",
                    "--disable-features=IsolateOrigins,site-per-process",
                ])

            self._browser = await self._playwright.chromium.launch(**launch_options)

            # Create default context
            await self._create_context("default")

            self._initialized = True
            logger.info("BrowserManager initialized")
            return True

        except ImportError:
            logger.error("Playwright not installed. Install: pip install playwright && playwright install chromium")
            return False
        except Exception as e:
            logger.exception(f"BrowserManager initialization failed: {e}")
            return False

    async def _create_context(self, context_id: str) -> Any:
        from playwright.async_api import BrowserContext

        context = await self._browser.new_context(
            viewport=self._config.viewport,
            user_agent=self._config.user_agent,
            proxy={"server": self._config.proxy} if self._config.proxy else None,
            accept_downloads=True,
            record_video_dir=self._config.downloads_path,
        )

        # Add stealth scripts
        if self._config.stealth:
            await context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                window.chrome = { runtime: {} };
                Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
                Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
            """)

        self._contexts[context_id] = context
        return context

    def get_context(self, context_id: str = "default") -> Any:
        return self._contexts.get(context_id)

    def get_page(self, page_id: str) -> Any:
        return self._pages.get(page_id)

    async def new_page(self, context_id: str = "default", page_id: str | None = None) -> str:
        context = self.get_context(context_id)
        if not context:
            await self._create_context(context_id)
            context = self.get_context(context_id)

        page = await context.new_page()
        pid = page_id or f"page_{len(self._pages)}"
        self._pages[pid] = page
        return pid

    async def close_page(self, page_id: str) -> bool:
        page = self._pages.pop(page_id, None)
        if page:
            await page.close()
            return True
        return False

    async def close_context(self, context_id: str) -> bool:
        for pid in list(self._pages.keys()):
            page = self._pages.get(pid)
            if page and page.context == self._contexts.get(context_id):
                await self.close_page(pid)

        context = self._contexts.pop(context_id, None)
        if context:
            await context.close()
            return True
        return False

    async def shutdown(self) -> None:
        for pid in list(self._pages.keys()):
            await self.close_page(pid)

        for cid in list(self._contexts.keys()):
            await self.close_context(cid)

        if self._browser:
            await self._browser.close()

        if self._playwright:
            await self._playwright.stop()

        self._initialized = False
        logger.info("BrowserManager shutdown")


# Global browser manager
_browser_manager = BrowserManager()


# =============================================================================
# BROWSER TOOLS
# =============================================================================

async def browser_init(params: dict) -> dict:
    config = BrowserConfig(
        headless=params.get("headless", True),
        viewport=params.get("viewport", {"width": 1280, "height": 720}),
        stealth=params.get("stealth", False),
        timeout=params.get("timeout", 30000),
    )

    success = await _browser_manager.initialize(config)
    return {"success": success, "message": "Browser initialized" if success else "Failed to initialize"}


async def browser_navigate(params: dict) -> dict:
    url = params.get("url", "")
    page_id = params.get("page_id", "default")
    wait_until = params.get("wait_until", "networkidle")
    timeout = params.get("timeout", 30000)

    if not url:
        return {"success": False, "error": "URL required"}

    page = _browser_manager.get_page(page_id)
    if not page:
        pid = await _browser_manager.new_page("default", page_id)
        page = _browser_manager.get_page(pid)

    try:
        response = await page.goto(url, wait_until=wait_until, timeout=timeout)
        return {
            "success": True,
            "url": page.url,
            "title": await page.title(),
            "status": response.status if response else None,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def browser_click(params: dict) -> dict:
    selector = params.get("selector", "")
    page_id = params.get("page_id", "default")
    button = params.get("button", "left")
    click_count = params.get("click_count", 1)
    delay = params.get("delay", 0)
    force = params.get("force", False)
    timeout = params.get("timeout", 30000)

    if not selector:
        return {"success": False, "error": "Selector required"}

    page = _browser_manager.get_page(page_id)
    if not page:
        return {"success": False, "error": "Page not found"}

    try:
        await page.click(selector, button=button, click_count=click_count,
                         delay=delay, force=force, timeout=timeout)
        return {"success": True, "message": f"Clicked {selector}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def browser_type(params: dict) -> dict:
    selector = params.get("selector", "")
    text = params.get("text", "")
    page_id = params.get("page_id", "default")
    delay = params.get("delay", 50)
    clear = params.get("clear", True)
    press_enter = params.get("press_enter", False)
    timeout = params.get("timeout", 30000)

    if not selector:
        return {"success": False, "error": "Selector required"}

    page = _browser_manager.get_page(page_id)
    if not page:
        return {"success": False, "error": "Page not found"}

    try:
        if clear:
            await page.fill(selector, "", timeout=timeout)

        await page.type(selector, text, delay=delay, timeout=timeout)

        if press_enter:
            await page.press(selector, "Enter", timeout=timeout)

        return {"success": True, "message": f"Typed into {selector}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def browser_hover(params: dict) -> dict:
    selector = params.get("selector", "")
    page_id = params.get("page_id", "default")
    timeout = params.get("timeout", 30000)

    page = _browser_manager.get_page(page_id)
    if not page:
        return {"success": False, "error": "Page not found"}

    try:
        await page.hover(selector, timeout=timeout)
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def browser_scroll(params: dict) -> dict:
    direction = params.get("direction", "down")
    pixels = params.get("pixels", 500)
    selector = params.get("selector")
    page_id = params.get("page_id", "default")

    page = _browser_manager.get_page(page_id)
    if not page:
        return {"success": False, "error": "Page not found"}

    try:
        if selector:
            element = await page.query_selector(selector)
            if not element:
                return {"success": False, "error": "Element not found"}

            if direction in ["up", "down"]:
                await element.evaluate(f"el => el.scrollBy(0, {pixels if direction == 'down' else -pixels})")
            else:
                await element.evaluate(f"el => el.scrollBy({pixels if direction == 'right' else -pixels}, 0)")
        else:
            if direction in ["up", "down"]:
                await page.evaluate(f"window.scrollBy(0, {pixels if direction == 'down' else -pixels})")
            elif direction == "top":
                await page.evaluate("window.scrollTo(0, 0)")
            elif direction == "bottom":
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            else:
                await page.evaluate(f"window.scrollBy({pixels if direction == 'right' else -pixels}, 0)")

        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def browser_wait(params: dict) -> dict:
    condition = params.get("condition", "load")
    selector = params.get("selector")
    timeout = params.get("timeout", 30000)
    page_id = params.get("page_id", "default")

    page = _browser_manager.get_page(page_id)
    if not page:
        return {"success": False, "error": "Page not found"}

    try:
        if condition == "selector_visible":
            await page.wait_for_selector(selector, state="visible", timeout=timeout)
        elif condition == "selector_hidden":
            await page.wait_for_selector(selector, state="hidden", timeout=timeout)
        elif condition == "selector_attached":
            await page.wait_for_selector(selector, state="attached", timeout=timeout)
        elif condition == "navigation":
            await page.wait_for_load_state("networkidle", timeout=timeout)
        elif condition == "load":
            await page.wait_for_load_state("load", timeout=timeout)
        elif condition == "domcontentloaded":
            await page.wait_for_load_state("domcontentloaded", timeout=timeout)
        elif condition == "networkidle":
            await page.wait_for_load_state("networkidle", timeout=timeout)
        else:
            return {"success": False, "error": f"Unknown condition: {condition}"}

        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def browser_extract(params: dict) -> dict:
    selector = params.get("selector", "")
    attribute = params.get("attribute", "text")
    multiple = params.get("multiple", False)
    page_id = params.get("page_id", "default")
    timeout = params.get("timeout", 30000)

    if not selector:
        return {"success": False, "error": "Selector required"}

    page = _browser_manager.get_page(page_id)
    if not page:
        return {"success": False, "error": "Page not found"}

    try:
        if multiple:
            elements = await page.query_selector_all(selector)
            results = []
            for el in elements:
                if attribute == "text":
                    val = await el.text_content()
                elif attribute == "html":
                    val = await el.inner_html()
                else:
                    val = await el.get_attribute(attribute)
                results.append(val)
            return {"success": True, "data": results, "count": len(results)}
        else:
            element = await page.query_selector(selector)
            if not element:
                return {"success": False, "error": "Element not found"}

            if attribute == "text":
                val = await element.text_content()
            elif attribute == "html":
                val = await element.inner_html()
            else:
                val = await element.get_attribute(attribute)

            return {"success": True, "data": val}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def browser_screenshot(params: dict) -> dict:
    output = params.get("output", "screenshot.png")
    page_id = params.get("page_id", "default")
    full_page = params.get("full_page", True)
    selector = params.get("selector")
    img_type = params.get("type", "png")
    quality = params.get("quality", 90)

    page = _browser_manager.get_page(page_id)
    if not page:
        return {"success": False, "error": "Page not found"}

    try:
        options = {
            "path": output,
            "full_page": full_page,
            "type": img_type,
        }

        if img_type == "jpeg":
            options["quality"] = quality

        if selector:
            element = await page.query_selector(selector)
            if not element:
                return {"success": False, "error": "Element not found"}
            await element.screenshot(**options)
        else:
            await page.screenshot(**options)

        return {"success": True, "path": output}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def browser_pdf(params: dict) -> dict:
    output = params.get("output", "page.pdf")
    page_id = params.get("page_id", "default")
    format = params.get("format", "A4")
    print_background = params.get("print_background", True)
    margin = params.get("margin", {"top": "1cm", "bottom": "1cm", "left": "1cm", "right": "1cm"})

    page = _browser_manager.get_page(page_id)
    if not page:
        return {"success": False, "error": "Page not found"}

    try:
        await page.pdf(
            path=output,
            format=format,
            print_background=print_background,
            margin=margin,
        )
        return {"success": True, "path": output}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def browser_evaluate(params: dict) -> dict:
    script = params.get("script", "")
    args = params.get("args", [])
    page_id = params.get("page_id", "default")

    if not script:
        return {"success": False, "error": "Script required"}

    page = _browser_manager.get_page(page_id)
    if not page:
        return {"success": False, "error": "Page not found"}

    try:
        result = await page.evaluate(script, *args)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def browser_new_tab(params: dict) -> dict:
    context_id = params.get("context_id", "default")
    page_id = params.get("page_id")
    url = params.get("url")

    try:
        pid = await _browser_manager.new_page(context_id, page_id)
        page = _browser_manager.get_page(pid)

        if url:
            await page.goto(url, wait_until="networkidle")

        return {"success": True, "page_id": pid, "url": page.url}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def browser_close_tab(params: dict) -> dict:
    page_id = params.get("page_id", "default")

    success = await _browser_manager.close_page(page_id)
    return {"success": success, "message": "Page closed" if success else "Page not found"}


async def browser_list_tabs(params: dict) -> dict:
    pages = []
    for pid, page in _browser_manager._pages.items():
        try:
            pages.append({
                "page_id": pid,
                "url": page.url,
                "title": await page.title(),
            })
        except Exception:
            pages.append({"page_id": pid, "url": "unknown", "title": "unknown"})

    return {"success": True, "pages": pages, "count": len(pages)}


async def browser_download(params: dict) -> dict:
    selector = params.get("selector", "")
    page_id = params.get("page_id", "default")
    save_as = params.get("save_as")
    timeout = params.get("timeout", 30000)

    page = _browser_manager.get_page(page_id)
    if not page:
        return {"success": False, "error": "Page not found"}

    try:
        async with page.expect_download(timeout=timeout) as download_info:
            await page.click(selector)

        download = await download_info.value

        if save_as:
            await download.save_as(save_as)
            path = save_as
        else:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=download.suggested_filename) as f:
                path = f.name
            await download.save_as(path)

        return {"success": True, "path": path, "filename": download.suggested_filename}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def browser_shutdown(params: dict) -> dict:
    await _browser_manager.shutdown()
    return {"success": True, "message": "Browser shutdown"}


# =============================================================================
# REGISTRATION
# =============================================================================

def register_browser_tools(registry: ToolRegistry) -> None:
    tools = [
        ("browser_init", browser_init, "Initialize browser", [
            ToolParameter("headless", "boolean", "Run headless", default=True),
            ToolParameter("viewport", "object", "Viewport size", default={"width": 1280, "height": 720}),
            ToolParameter("stealth", "boolean", "Enable stealth mode", default=False),
            ToolParameter("timeout", "integer", "Default timeout ms", default=30000),
        ], {"requires_approval": False, "sandboxed": True}),

        ("browser_navigate", browser_navigate, "Navigate to URL", [
            ToolParameter("url", "string", "Target URL", required=True),
            ToolParameter("page_id", "string", "Page ID", default="default"),
            ToolParameter("wait_until", "string", "Wait condition: load, domcontentloaded, networkidle", default="networkidle"),
            ToolParameter("timeout", "integer", "Timeout ms", default=30000),
        ], {"requires_approval": False, "sandboxed": True}),

        ("browser_click", browser_click, "Click element", [
            ToolParameter("selector", "string", "CSS/XPath selector", required=True),
            ToolParameter("page_id", "string", "Page ID", default="default"),
            ToolParameter("button", "string", "Mouse button: left, right, middle", default="left"),
            ToolParameter("click_count", "integer", "Number of clicks", default=1),
            ToolParameter("delay", "integer", "Delay between clicks ms", default=0),
            ToolParameter("force", "boolean", "Force click", default=False),
            ToolParameter("timeout", "integer", "Timeout ms", default=30000),
        ], {"requires_approval": False, "sandboxed": True}),

        ("browser_type", browser_type, "Type text into element", [
            ToolParameter("selector", "string", "CSS/XPath selector", required=True),
            ToolParameter("text", "string", "Text to type", required=True),
            ToolParameter("page_id", "string", "Page ID", default="default"),
            ToolParameter("delay", "integer", "Delay between keystrokes ms", default=50),
            ToolParameter("clear", "boolean", "Clear field first", default=True),
            ToolParameter("press_enter", "boolean", "Press Enter after", default=False),
            ToolParameter("timeout", "integer", "Timeout ms", default=30000),
        ], {"requires_approval": False, "sandboxed": True}),

        ("browser_hover", browser_hover, "Hover over element", [
            ToolParameter("selector", "string", "CSS/XPath selector", required=True),
            ToolParameter("page_id", "string", "Page ID", default="default"),
            ToolParameter("timeout", "integer", "Timeout ms", default=30000),
        ], {"requires_approval": False, "sandboxed": True}),

        ("browser_scroll", browser_scroll, "Scroll page", [
            ToolParameter("direction", "string", "Direction: up, down, left, right, top, bottom", default="down"),
            ToolParameter("pixels", "integer", "Pixels to scroll", default=500),
            ToolParameter("selector", "string", "Scroll within element"),
            ToolParameter("page_id", "string", "Page ID", default="default"),
        ], {"requires_approval": False, "sandboxed": True}),

        ("browser_wait", browser_wait, "Wait for condition", [
            ToolParameter("condition", "string", "Condition: selector_visible, selector_hidden, navigation, load, networkidle", default="load"),
            ToolParameter("selector", "string", "Selector for visibility"),
            ToolParameter("timeout", "integer", "Timeout ms", default=30000),
            ToolParameter("page_id", "string", "Page ID", default="default"),
        ], {"requires_approval": False, "sandboxed": True}),

        ("browser_extract", browser_extract, "Extract data from page", [
            ToolParameter("selector", "string", "CSS selector", required=True),
            ToolParameter("attribute", "string", "Attribute: text, html, href, src, value", default="text"),
            ToolParameter("multiple", "boolean", "Return all matches", default=False),
            ToolParameter("page_id", "string", "Page ID", default="default"),
            ToolParameter("timeout", "integer", "Timeout ms", default=30000),
        ], {"requires_approval": False, "sandboxed": True}),

        ("browser_screenshot", browser_screenshot, "Take screenshot", [
            ToolParameter("output", "string", "Output path", default="screenshot.png"),
            ToolParameter("page_id", "string", "Page ID", default="default"),
            ToolParameter("full_page", "boolean", "Full page screenshot", default=True),
            ToolParameter("selector", "string", "Screenshot element"),
            ToolParameter("type", "string", "Image type: png, jpeg", default="png"),
            ToolParameter("quality", "integer", "JPEG quality", default=90),
        ], {"requires_approval": False, "sandboxed": True}),

        ("browser_pdf", browser_pdf, "Save page as PDF", [
            ToolParameter("output", "string", "Output path", default="page.pdf"),
            ToolParameter("page_id", "string", "Page ID", default="default"),
            ToolParameter("format", "string", "Paper format: A4, Letter", default="A4"),
            ToolParameter("print_background", "boolean", "Include backgrounds", default=True),
            ToolParameter("margin", "object", "Margins", default={"top": "1cm", "bottom": "1cm", "left": "1cm", "right": "1cm"}),
        ], {"requires_approval": False, "sandboxed": True}),

        ("browser_evaluate", browser_evaluate, "Execute JavaScript", [
            ToolParameter("script", "string", "JavaScript code", required=True),
            ToolParameter("args", "array", "Arguments to script"),
            ToolParameter("page_id", "string", "Page ID", default="default"),
        ], {"requires_approval": True, "sandboxed": True}),

        ("browser_new_tab", browser_new_tab, "Create new tab", [
            ToolParameter("context_id", "string", "Browser context", default="default"),
            ToolParameter("page_id", "string", "Custom page ID"),
            ToolParameter("url", "string", "Initial URL"),
        ], {"requires_approval": False, "sandboxed": True}),

        ("browser_close_tab", browser_close_tab, "Close tab", [
            ToolParameter("page_id", "string", "Page ID", default="default"),
        ], {"requires_approval": False, "sandboxed": True}),

        ("browser_list_tabs", browser_list_tabs, "List open tabs", [], {"requires_approval": False, "sandboxed": True}),

        ("browser_download", browser_download, "Handle file download", [
            ToolParameter("selector", "string", "Download trigger selector", required=True),
            ToolParameter("page_id", "string", "Page ID", default="default"),
            ToolParameter("save_as", "string", "Save path"),
            ToolParameter("timeout", "integer", "Timeout ms", default=30000),
        ], {"requires_approval": False, "sandboxed": True}),

        ("browser_shutdown", browser_shutdown, "Shutdown browser", [], {"requires_approval": False, "sandboxed": True}),
    ]

    for name, func, desc, params, extra in tools:
        metadata = ToolMetadata(
            name=name,
            description=desc,
            category="browser",
            parameters=params,
            returns={"type": "object"},
            sandboxed=extra.get("sandboxed", True),
            requires_approval=extra.get("requires_approval", False),
            tags=["browser", "automation", "web", "playwright"],
        )
        registry.register(name, func, metadata)

    logger.info("Registered browser tools")


__all__ = [
    "register_browser_tools",
    "BrowserManager",
    "BrowserConfig",
]