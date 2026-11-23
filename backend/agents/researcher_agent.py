"""
JARVIS OMEGA - Researcher Agent
Specialist agent for web search, document analysis, fact extraction, and synthesis.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from agents.base_agent import BaseAgent, AgentConfig, AgentContext
from tools.tool_registry import ToolRegistry, ToolMetadata, ToolParameter
from config import get_settings, settings

logger = logging.getLogger("jarvis.agents.researcher")


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class SearchResult:
    """Single search result."""
    title: str
    url: str
    snippet: str
    source: str
    timestamp: float = field(default_factory=time.time)
    relevance_score: float = 0.0


@dataclass
class Document:
    """Analyzed document."""
    doc_id: str
    source: str  # url, file path, etc.
    title: str
    content: str
    summary: str = ""
    key_points: list[str] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ResearchReport:
    """Complete research report."""
    query: str
    executive_summary: str
    sections: list[dict]  # {"title": "", "content": "", "sources": []}
    sources: list[SearchResult]
    confidence: float
    generated_at: float = field(default_factory=time.time)


# =============================================================================
# RESEARCHER AGENT
# =============================================================================

class ResearcherAgent(BaseAgent):
    """
    Researcher Agent - Information gathering, analysis, synthesis.
    The "brain" for knowledge work.
    """

    def __init__(
        self,
        event_bus: Any = None,
        memory_manager: Any = None,
        tool_registry: Any = None,
        llm_client: Any = None,
    ):
        agent_config = AgentConfig(
            name="researcher",
            role="researcher",
            description="Web search, document analysis, fact extraction, multi-source synthesis, report generation",
            model="researcher",
            temperature=0.4,
            max_tokens=8192,
            system_prompt=self._get_system_prompt(),
            tools=[
                "web_search", "deep_search", "fetch_page", "extract_content",
                "analyze_document", "summarize", "fact_check", "compare_sources",
                "generate_report", "cite_sources",
            ],
            can_delegate=False,
            can_handoff=True,
            stream_response=True,
            timeout=180.0,
        )

        super().__init__(
            config=agent_config,
            event_bus=event_bus,
            memory_manager=memory_manager,
            tool_registry=tool_registry,
            llm_client=llm_client,
        )

        # Search APIs
        self._session: aiohttp.ClientSession | None = None
        self._search_cache: dict[str, list[SearchResult]] = {}

        # Register tools
        if self.tool_registry:
            self._register_tools()

    def _get_system_prompt(self) -> str:
        return """You are the Researcher Agent - VEGA's knowledge specialist. You find, analyze, and synthesize information from multiple sources.

Capabilities:
- WEB SEARCH: DuckDuckGo, Google, Bing, Brave, academic sources
- DOCUMENT ANALYSIS: PDF, HTML, text, markdown - extraction, summarization, key points
- FACT CHECKING: Cross-reference multiple sources, verify claims
- SYNTHESIS: Combine findings into coherent reports with citations
- COMPETITIVE INTELLIGENCE: Market research, tech trends, company analysis

Guidelines:
1. ALWAYS cite sources with URLs
2. Distinguish between facts, opinions, and speculation
3. Rate confidence in findings (High/Medium/Low)
3. Flag conflicting information
4. Prefer primary sources over secondary
5. Note publication dates for time-sensitive info
6. Structure reports: Executive Summary → Sections → Sources

Output Format:
- Executive Summary (2-3 sentences)
- Key Findings (bulleted, with citations)
- Detailed Analysis (sections with evidence)
- Sources (numbered with URLs and access dates)
- Confidence Assessment""".

    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------

    async def _agent_initialize(self) -> None:
        """Initialize HTTP session and search clients."""
        import aiohttp
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": "VEGA/1.0 (compatible; Research Bot)"},
        )

        # Initialize search providers
        self._search_providers: list[SearchProvider] = [
            DuckDuckGoSearch(self._session),
        ]

        # Add Brave if API key available
        brave_key = get_settings().get("brave_api_key") if hasattr(get_settings(), 'brave_api_key') else None
        if brave_key:
            self._search_providers.append(BraveSearch(self._session, brave_key))

        logger.info("ResearcherAgent initialized")

    async def _agent_shutdown(self) -> None:
        if self._session:
            await self._session.close()

    # -------------------------------------------------------------------------
    # TOOL REGISTRATION
    # -------------------------------------------------------------------------

    def _register_tools(self) -> None:
        tools = {
            "web_search": self._tool_web_search,
            "deep_search": self._tool_deep_search,
            "fetch_page": self._tool_fetch_page,
            "extract_content": self._tool_extract_content,
            "analyze_document": self._tool_analyze_document,
            "summarize": self._tool_summarize,
            "fact_check": self._tool_fact_check,
            "compare_sources": self._tool_compare_sources,
            "generate_report": self._tool_generate_report,
            "cite_sources": self._tool_cite_sources,
        }

        for name, func in tools.items():
            self.tool_registry.register(name, func)

    # -------------------------------------------------------------------------
    # TASK PROCESSING
    # -------------------------------------------------------------------------

    async def process_task(
        self,
        task: str,
        context: AgentContext,
    ) -> Any:
        """Process research task."""
        task_lower = task.lower()

        if any(kw in task_lower for kw in ["report", "research paper", "comprehensive"]):
            return await self._deep_research(task, context)
        elif any(kw in task_lower for kw in ["search", "find", "look up"]):
            return await self._quick_search(task, context)
        elif any(kw in task_lower for kw in ["analyze", "summarize", "extract"]):
            return await self._analyze_content(task, context)
        elif any(kw in task_lower for kw in ["fact check", "verify", "true or false"]):
            return await self._fact_check_claim(task, context)
        else:
            # Default: quick search + synthesis
            return await self._quick_search(task, context)

    async def process_task_stream(
        self,
        task: str,
        context: AgentContext,
    ) -> AsyncGenerator[str, None]:
        """Stream research progress."""
        yield f"🔬 Researcher starting: {task[:100]}...\n\n"

        if "report" in task.lower() or "comprehensive" in task.lower():
            async for chunk in self._deep_research_stream(task, context):
                yield chunk
        else:
            async for chunk in self._quick_search_stream(task, context):
                yield chunk

    # -------------------------------------------------------------------------
    # SEARCH IMPLEMENTATIONS
    # -------------------------------------------------------------------------

    async def _quick_search(self, query: str, context: AgentContext) -> str:
        """Quick web search and synthesis."""
        results = await self._search_web(query, max_results=5)

        if not results:
            return "No results found."

        synthesis = await self._synthesize_results(query, results)
        return synthesis

    async def _quick_search_stream(self, query: str, context: AgentContext) -> AsyncGenerator[str, None]:
        """Stream quick search."""
        yield f"🔍 Searching for: {query}\n\n"

        results = await self._search_web(query, max_results=5)

        if not results:
            yield "❌ No results found.\n"
            return

        yield f"📄 Found {len(results)} results. Synthesizing...\n\n"

        async for chunk in self._synthesize_results_stream(query, results):
            yield chunk

    async def _deep_research(self, query: str, context: AgentContext) -> ResearchReport:
        """Comprehensive multi-source research."""
        # Multi-round search
        all_results = []

        # Round 1: Broad search
        results1 = await self._search_web(query, max_results=10)
        all_results.extend(results1)

        # Round 2: Specific aspects
        aspects = await self._identify_research_aspects(query)
        for aspect in aspects[:3]:
            aspect_results = await self._search_web(f"{query} {aspect}", max_results=5)
            all_results.extend(aspect_results)

        # Deduplicate
        unique_results = self._deduplicate_results(all_results)

        # Fetch full content for top results
        documents = await self._fetch_documents(unique_results[:10])

        # Analyze documents
        analyzed = await self._analyze_documents(documents, query)

        # Generate report
        report = await self._generate_research_report(query, analyzed, unique_results)

        return report

    async def _deep_research_stream(self, query: str, context: AgentContext) -> AsyncGenerator[str, None]:
        """Stream deep research."""
        yield f"📚 **Deep Research: {query}**\n\n"

        # Phase 1: Broad search
        yield "🔍 **Phase 1**: Broad search...\n"
        results1 = await self._search_web(query, max_results=10)
        yield f"   Found {len(results1)} results\n\n"

        # Phase 2: Aspect identification
        yield "🧭 **Phase 2**: Identifying research aspects...\n"
        aspects = await self._identify_research_aspects(query)
        for a in aspects[:3]:
            yield f"   - {a}\n"
        yield "\n"

        # Phase 3: Targeted searches
        yield "🎯 **Phase 3**: Targeted searches...\n"
        all_results = list(results1)
        for aspect in aspects[:3]:
            aspect_results = await self._search_web(f"{query} {aspect}", max_results=5)
            all_results.extend(aspect_results)
            yield f"   '{aspect}': {len(aspect_results)} results\n"

        yield "\n📄 **Phase 4**: Fetching & analyzing documents...\n"
        unique_results = self._deduplicate_results(all_results)
        documents = await self._fetch_documents(unique_results[:10])
        analyzed = await self._analyze_documents(documents, query)

        yield f"   Analyzed {len(analyzed)} documents\n\n"

        # Phase 5: Report generation
        yield "📝 **Phase 5**: Generating report...\n\n"
        report = await self._generate_research_report(query, analyzed, unique_results)

        # Stream report
        yield f"# {report.query}\n\n"
        yield f"## Executive Summary\n{report.executive_summary}\n\n"

        for section in report.sections:
            yield f"## {section['title']}\n{section['content']}\n\n"

        yield f"## Sources ({len(report.sources)})\n"
        for i, src in enumerate(report.sources, 1):
            yield f"{i}. [{src.title}]({src.url}) - {src.source}\n"

        yield f"\n**Confidence**: {report.confidence:.0%}\n"

    # -------------------------------------------------------------------------
    # SEARCH BACKENDS
    # -------------------------------------------------------------------------

    async def _search_web(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """Search web using multiple backends."""
        # Check cache
        cache_key = hashlib.md5(f"{query}:{max_results}".encode()).hexdigest()
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]

        results = []

        # Try DuckDuckGo HTML scraping (no API key needed)
        ddg_results = await self._search_duckduckgo(query, max_results)
        results.extend(ddg_results)

        # Try Brave Search if API key available
        brave_key = get_settings().get("brave_api_key") if hasattr(get_settings(), 'brave_api_key') else None
        if brave_key:
            brave_results = await self._search_brave(query, max_results, brave_key)
            results.extend(brave_results)

        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        # Cache
        self._search_cache[cache_key] = results[:max_results]

        return results[:max_results]

    async def _search_duckduckgo(self, query: str, max_results: int) -> list[SearchResult]:
        """Search DuckDuckGo via HTML."""
        if not self._session:
            return []

        try:
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            headers = {"User-Agent": "Mozilla/5.0 (compatible; VEGA/1.0)"}

            async with self._session.get(url, headers=headers) as resp:
                html = await resp.text()

            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')

            results = []
            for result in soup.select('.result__body')[:max_results]:
                title_el = result.select_one('.result__title')
                snippet_el = result.select_one('.result__snippet')
                url_el = result.select_one('.result__url')

                if title_el:
                    title = title_el.get_text(strip=True)
                    snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                    url = url_el.get_text(strip=True) if url_el else ""

                    results.append(SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source="duckduckgo",
                        relevance_score=0.8,
                    ))

            return results
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []

    async def _search_brave(self, query: str, max_results: int, api_key: str) -> list[SearchResult]:
        """Search Brave Search API."""
        if not self._session:
            return []

        try:
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {"X-Subscription-Token": api_key}
            params = {"q": query, "count": max_results}

            async with self._session.get(url, headers=headers, params=params) as resp:
                data = await resp.json()

            results = []
            for item in data.get("web", {}).get("results", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("description", ""),
                    source="brave",
                    relevance_score=0.9,
                ))

            return results
        except Exception as e:
            logger.error(f"Brave search error: {e}")
            return []

    # -------------------------------------------------------------------------
    # CONTENT FETCHING & ANALYSIS
    # -------------------------------------------------------------------------

    async def _fetch_page(self, url: str) -> str | None:
        """Fetch and extract main content from URL."""
        if not self._session:
            return None

        try:
            async with self._session.get(url, timeout=15) as resp:
                if resp.status != 200:
                    return None
                html = await resp.text()

            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')

            # Remove script/style
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                tag.decompose()

            # Get text
            text = soup.get_text(separator='\n', strip=True)
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            return '\n'.join(lines[:500])  # Limit length

        except Exception as e:
            logger.error(f"Fetch page error for {url}: {e}")
            return None

    async def _fetch_documents(self, results: list[SearchResult]) -> list[Document]:
        """Fetch full content for search results."""
        documents = []

        for result in results[:10]:  # Limit concurrent fetches
            content = await self._fetch_page(result.url)
            if content:
                doc = Document(
                    doc_id=str(uuid.uuid4()),
                    source=result.url,
                    title=result.title,
                    content=content,
                    metadata={
                        "search_snippet": result.snippet,
                        "search_source": result.source,
                    },
                )
                documents.append(doc)

        return documents

    async def _analyze_documents(self, documents: list[Document], query: str) -> list[Document]:
        """Analyze documents with LLM."""
        if not self.llm_client or not documents:
            return documents

        analyzed = []

        for doc in documents:
            prompt = f"""Analyze this document for the query: "{query}"

Document: {doc.title}
Source: {doc.source}
Content: {doc.content[:3000]}

Extract:
1. Summary (2-3 sentences)
2. Key points (bullet list)
3. Named entities (people, orgs, dates, locations)
4. Relevance to query (High/Medium/Low)
5. Confidence in accuracy

Return JSON."""

            try:
                response = await self._call_llm([
                    {"role": "system", "content": "You are an expert analyst. Extract structured information."},
                    {"role": "user", "content": prompt},
                ])

                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                    doc.summary = analysis.get("summary", "")
                    doc.key_points = analysis.get("key_points", [])
                    doc.entities = analysis.get("entities", [])
                    doc.metadata["relevance"] = analysis.get("relevance", "Medium")
                    doc.metadata["confidence"] = analysis.get("confidence", "Medium")

            except Exception as e:
                logger.error(f"Document analysis error: {e}")

            analyzed.append(doc)

        return analyzed

    # -------------------------------------------------------------------------
    # SYNTHESIS & REPORTING
    # -------------------------------------------------------------------------

    async def _synthesize_results(self, query: str, results: list[SearchResult]) -> str:
        """Synthesize search results into answer."""
        if not self.llm_client:
            return self._format_results_simple(results)

        prompt = f"""Synthesize these search results to answer: "{query}"

Results:
{json.dumps([{"title": r.title, "url": r.url, "snippet": r.snippet, "source": r.source} for r in results], indent=2)}

Provide a clear, well-structured answer with inline citations [1], [2], etc.
Include a Sources section at the end."""

        response = await self._call_llm([
            {"role": "system", "content": "You are a research synthesizer. Cite sources inline."},
            {"role": "user", "content": prompt},
        ])

        return response

    async def _synthesize_results_stream(
        self,
        query: str,
        results: list[SearchResult],
    ) -> AsyncGenerator[str, None]:
        """Stream synthesis."""
        if not self.llm_client:
            for i, r in enumerate(results, 1):
                yield f"[{i}] **{r.title}** ({r.source})\n{r.snippet}\n{r.url}\n\n"
            return

        prompt = f"""Synthesize these search results to answer: "{query}"

Results:
{json.dumps([{"title": r.title, "url": r.url, "snippet": r.snippet, "source": r.source} for r in results], indent=2)}

Provide a clear, well-structured answer with inline citations [1], [2], etc.
Include a Sources section at the end."""

        async for chunk in self._call_llm_stream([
            {"role": "system", "content": "You are a research synthesizer. Cite sources inline."},
            {"role": "user", "content": prompt},
        ]):
            yield chunk

    def _format_results_simple(self, results: list[SearchResult]) -> str:
        """Simple formatting without LLM."""
        lines = [f"Found {len(results)} results for your query:\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"**[{i}] {r.title}** ({r.source})")
            lines.append(f"   {r.snippet[:200]}...")
            lines.append(f"   🔗 {r.url}\n")
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # HELPER METHODS
    # -------------------------------------------------------------------------

    async def _identify_research_aspects(self, query: str) -> list[str]:
        """Identify key aspects to research."""
        if not self.llm_client:
            return ["overview", "details", "examples", "comparison"]

        prompt = f"""For the research query: "{query}"
Identify 3-5 specific aspects or angles to investigate.
Return as JSON list of strings."""

        try:
            response = await self._call_llm([
                {"role": "system", "content": "Identify research aspects as JSON array."},
                {"role": "user", "content": prompt},
            ])
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass

        return ["overview", "details", "examples", "comparison"]

    def _deduplicate_results(self, results: list[SearchResult]) -> list[SearchResult]:
        """Remove duplicate URLs."""
        seen = set()
        unique = []
        for r in results:
            if r.url not in seen:
                seen.add(r.url)
                unique.append(r)
        return unique

    async def _generate_research_report(
        self,
        query: str,
        documents: list[Document],
        sources: list[SearchResult],
    ) -> ResearchReport:
        """Generate formal research report."""
        if not self.llm_client:
            return ResearchReport(
                query=query,
                executive_summary=f"Research on {query} - {len(documents)} documents analyzed",
                sections=[],
                sources=sources,
                confidence=0.5,
            )

        prompt = f"""Generate a comprehensive research report for: "{query}"

Analyzed Documents:
{json.dumps([{"title": d.title, "source": d.source, "summary": d.summary, "key_points": d.key_points, "relevance": d.metadata.get("relevance")} for d in documents], indent=2)}

Structure:
1. Executive Summary (3-4 sentences)
2. Sections (3-5 sections with titles and content)
3. Sources (already provided)

Return JSON with: executive_summary, sections[{{title, content, sources}}]"""

        try:
            response = await self._call_llm([
                {"role": "system", "content": "Generate structured research report as JSON."},
                {"role": "user", "content": prompt},
            ])

            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                report_data = json.loads(json_match.group())

                return ResearchReport(
                    query=query,
                    executive_summary=report_data.get("executive_summary", ""),
                    sections=report_data.get("sections", []),
                    sources=sources,
                    confidence=0.8,
                )
        except Exception as e:
            logger.error(f"Report generation error: {e}")

        return ResearchReport(
            query=query,
            executive_summary="Report generation failed",
            sections=[],
            sources=sources,
            confidence=0.3,
        )

    # -------------------------------------------------------------------------
    # TOOL IMPLEMENTATIONS
    # -------------------------------------------------------------------------

    async def _tool_web_search(self, params: dict) -> dict:
        query = params.get("query", "")
        max_results = params.get("max_results", 10)

        results = await self._search_web(query, max_results)
        return {
            "results": [
                {"title": r.title, "url": r.url, "snippet": r.snippet, "source": r.source}
                for r in results
            ]
        }

    async def _tool_deep_search(self, params: dict) -> dict:
        query = params.get("query", "")
        report = await self._deep_research(query, AgentContext(conversation_id="tool"))

        return {
            "query": report.query,
            "executive_summary": report.executive_summary,
            "sections": report.sections,
            "sources": [{"title": s.title, "url": s.url, "source": s.source} for s in report.sources],
            "confidence": report.confidence,
        }

    async def _tool_fetch_page(self, params: dict) -> dict:
        url = params.get("url", "")
        content = await self._fetch_page(url)
        return {"url": url, "content": content, "success": content is not None}

    async def _tool_extract_content(self, params: dict) -> dict:
        url = params.get("url", "")
        selector = params.get("selector", "")

        content = await self._fetch_page(url)
        if not content:
            return {"success": False, "error": "Failed to fetch"}

        if selector:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            elements = soup.select(selector)
            return {"success": True, "content": [el.get_text(strip=True) for el in elements]}

        return {"success": True, "content": content}

    async def _tool_analyze_document(self, params: dict) -> dict:
        content = params.get("content", "")
        query = params.get("query", "Analyze this document")

        doc = Document(
            doc_id=str(uuid.uuid4()),
            source=params.get("source", "tool_input"),
            title=params.get("title", "Document"),
            content=content,
        )

        analyzed = await self._analyze_documents([doc], query)
        doc = analyzed[0]

        return {
            "summary": doc.summary,
            "key_points": doc.key_points,
            "entities": doc.entities,
            "relevance": doc.metadata.get("relevance"),
            "confidence": doc.metadata.get("confidence"),
        }

    async def _tool_summarize(self, params: dict) -> dict:
        content = params.get("content", "")
        max_length = params.get("max_length", 200)

        if not self.llm_client:
            return {"summary": content[:max_length] + "..."}

        prompt = f"Summarize in {max_length} words:\n\n{content[:3000]}"
        summary = await self._call_llm([
            {"role": "system", "content": "Summarize concisely."},
            {"role": "user", "content": prompt},
        ])

        return {"summary": summary}

    async def _tool_fact_check(self, params: dict) -> dict:
        claim = params.get("claim", "")

        results = await self._search_web(f"fact check: {claim}", max_results=5)

        if self.llm_client:
            prompt = f"""Fact-check this claim: "{claim}"

Evidence:
{json.dumps([{"title": r.title, "snippet": r.snippet, "url": r.url} for r in results], indent=2)}

Verdict: True / False / Partially True / Unverifiable
Confidence: High/Medium/Low
Explanation with citations."""

            analysis = await self._call_llm([
                {"role": "system", "content": "You are a fact-checker. Be precise."},
                {"role": "user", "content": prompt},
            ])

            return {"claim": claim, "analysis": analysis, "sources": len(results)}

        return {"claim": claim, "analysis": "LLM not available", "sources": len(results)}

    async def _tool_compare_sources(self, params: dict) -> dict:
        urls = params.get("urls", [])
        question = params.get("question", "Compare these sources")

        documents = []
        for url in urls[:5]:
            content = await self._fetch_page(url)
            if content:
                documents.append(Document(
                    doc_id=str(uuid.uuid4()),
                    source=url,
                    title=url,
                    content=content,
                ))

        analyzed = await self._analyze_documents(documents, question)

        return {
            "question": question,
            "sources": [
                {"url": d.source, "summary": d.summary, "key_points": d.key_points}
                for d in analyzed
            ],
        }

    async def _tool_generate_report(self, params: dict) -> dict:
        query = params.get("query", "")
        report = await self._deep_research(query, AgentContext(conversation_id="tool"))

        return {
            "query": report.query,
            "executive_summary": report.executive_summary,
            "sections": report.sections,
            "sources": [{"title": s.title, "url": s.url} for s in report.sources],
            "confidence": report.confidence,
        }

    async def _tool_cite_sources(self, params: dict) -> dict:
        sources = params.get("sources", [])
        style = params.get("style", "apa")

        return {"formatted": "Citation formatting not fully implemented", "style": style}

    # -------------------------------------------------------------------------
    # HELPER METHODS
    # -------------------------------------------------------------------------

    async def _fetch_page(self, url: str) -> str | None:
        """Fetch and extract main content from URL."""
        if not self._session:
            return None

        try:
            async with self._session.get(url, timeout=15) as resp:
                if resp.status != 200:
                    return None
                html = await resp.text()

            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')

            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                tag.decompose()

            text = soup.get_text(separator='\n', strip=True)
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            return '\n'.join(lines[:500])

        except Exception as e:
            logger.error(f"Fetch page error for {url}: {e}")
            return None

    async def _identify_research_aspects(self, query: str) -> list[str]:
        """Identify key aspects to research."""
        if not self.llm_client:
            return ["overview", "details", "examples", "comparison"]

        prompt = f"""For the research query: "{query}"
Identify 3-5 specific aspects or angles to investigate.
Return as JSON list of strings."""

        try:
            response = await self._call_llm([
                {"role": "system", "content": "Identify research aspects as JSON array."},
                {"role": "user", "content": prompt},
            ])
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass

        return ["overview", "details", "examples", "comparison"]

    def _deduplicate_results(self, results: list[SearchResult]) -> list[SearchResult]:
        seen = set()
        unique = []
        for r in results:
            if r.url not in seen:
                seen.add(r.url)
                unique.append(r)
        return unique

    # -------------------------------------------------------------------------
    # EXPORTS
    # -------------------------------------------------------------------------

__all__ = [
    "ResearcherAgent",
    "SearchResult",
    "Document",
    "ResearchReport",
]