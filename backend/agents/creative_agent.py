"""
JARVIS OMEGA - Creative Agent
Specialist agent for content generation: images, video, audio, text, presentations.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import aiohttp

from agents.base_agent import BaseAgent, AgentConfig, AgentContext
from tools.tool_registry import ToolRegistry, ToolMetadata, ToolParameter
from config import get_settings, settings

logger = logging.getLogger("jarvis.agents.creative")


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class GenerationRequest:
    """Content generation request."""
    type: str  # image, video, audio, text, presentation
    prompt: str
    style: str | None = None
    parameters: dict = field(default_factory=dict)
    negative_prompt: str | None = None
    seed: int | None = None
    width: int = 1024
    height: int = 1024
    num_outputs: int = 1


@dataclass
class GenerationResult:
    """Generation result."""
    success: bool
    type: str
    files: list[str] = field(default_factory=list)  # Local file paths
    urls: list[str] = field(default_factory=list)   # Remote URLs
    metadata: dict = field(default_factory=dict)
    error: str | None = None
    generation_time: float = 0.0


# =============================================================================
# CREATIVE AGENT
# =============================================================================

class CreativeAgent(BaseAgent):
    """
    Creative Agent - Generates images, video, audio, text, presentations.
    The "artist" of VEGA.
    """

    def __init__(
        self,
        event_bus: Any = None,
        memory_manager: Any = None,
        tool_registry: Any = None,
        llm_client: Any = None,
    ):
        agent_config = AgentConfig(
            name="creative",
            role="creative",
            description="Image/video/audio generation, content writing, presentation design, creative coding",
            model="creative",
            temperature=0.8,  # Higher for creativity
            max_tokens=8192,
            system_prompt=self._get_system_prompt(),
            tools=[
                "generate_image", "edit_image", "upscale_image", "image_variations",
                "generate_video", "animate_image",
                "generate_audio", "generate_music", "text_to_speech",
                "write_content", "write_code", "write_script",
                "create_presentation", "design_slide",
                "brainstorm", "prompt_enhance",
            ],
            can_delegate=False,
            can_handoff=True,
            stream_response=True,
            timeout=300.0,  # Long for video/audio
        )

        super().__init__(
            config=agent_config,
            event_bus=event_bus,
            memory_manager=memory_manager,
            tool_registry=tool_registry,
            llm_client=llm_client,
        )

        # Output directory
        self._output_dir = Path.home() / ".vega" / "creative"
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # API clients
        self._session: aiohttp.ClientSession | None = None

        # Register tools
        if self.tool_registry:
            self._register_tools()

    def _get_system_prompt(self) -> str:
        return """You are the Creative Agent - VEGA's artist and content creator. You generate and manipulate multimedia content.

Capabilities:
- IMAGES: DALL-E 3, Midjourney, Stable Diffusion XL, SDXL Turbo, FLUX
- VIDEO: Runway Gen-2, Pika, Stable Video Diffusion, Luma Dream Machine
- AUDIO: Suno, Udio, MusicGen, AudioLDM, ElevenLabs TTS
- TEXT: Articles, stories, scripts, poetry, marketing copy, technical docs
- CODE: Creative coding, shaders, visualizations, interactive art
- PRESENTATIONS: Slides, pitch decks, reports, infographics

Guidelines:
1. Enhance prompts for better results - add style, lighting, composition details
2. Provide multiple variations when possible
3. Save all outputs to organized directory structure
4. Return file paths and metadata
5. Respect copyright - don't generate trademarked characters/logos
6. For video/audio, warn about long generation times
7. Suggest iterative refinement

Output: Always provide local file paths, generation parameters, and suggestions for improvement.""".

    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------

    async def _agent_initialize(self) -> None:
        """Initialize HTTP session."""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=600),  # 10 min for video
        )
        logger.info("CreativeAgent initialized")

    async def _agent_shutdown(self) -> None:
        if self._session:
            await self._session.close()

    # -------------------------------------------------------------------------
    # TOOL REGISTRATION
    # -------------------------------------------------------------------------

    def _register_tools(self) -> None:
        tools = [
            # Images
            ("generate_image", self._tool_generate_image, "Generate images", [
                ToolParameter("prompt", "string", "Image description", required=True),
                ToolParameter("style", "string", "Art style"),
                ToolParameter("width", "integer", "Width", default=1024),
                ToolParameter("height", "integer", "Height", default=1024),
                ToolParameter("num_outputs", "integer", "Number of images", default=1),
            ]),
            ("edit_image", self._tool_edit_image, "Edit/modify existing images", [
                ToolParameter("image_path", "string", "Path to image", required=True),
                ToolParameter("prompt", "string", "Edit instruction", required=True),
            ]),
            ("upscale_image", self._tool_upscale_image, "Upscale image", [
                ToolParameter("image_path", "string", "Path to image", required=True),
                ToolParameter("scale", "integer", "Upscale factor", default=2),
            ]),
            ("image_variations", self._tool_image_variations, "Create image variations", [
                ToolParameter("image_path", "string", "Path to image", required=True),
                ToolParameter("count", "integer", "Number of variations", default=4),
            ]),
            # Video
            ("generate_video", self._tool_generate_video, "Generate video from text/image", [
                ToolParameter("prompt", "string", "Video description", required=True),
                ToolParameter("image_path", "string", "Reference image (for img2vid)"),
                ToolParameter("duration", "integer", "Duration in seconds", default=5),
            ]),
            ("animate_image", self._tool_animate_image, "Animate static image", [
                ToolParameter("image_path", "string", "Path to image", required=True),
                ToolParameter("motion_bucket_id", "integer", "Motion amount", default=127),
            ]),
            # Audio
            ("generate_audio", self._tool_generate_audio, "Generate sound effects", [
                ToolParameter("prompt", "string", "Sound description", required=True),
                ToolParameter("duration", "integer", "Duration in seconds", default=10),
            ]),
            ("generate_music", self._tool_generate_music, "Generate music", [
                ToolParameter("prompt", "string", "Music description", required=True),
                ToolParameter("duration", "integer", "Duration in seconds", default=30),
            ]),
            ("text_to_speech", self._tool_text_to_speech, "Convert text to speech", [
                ToolParameter("text", "string", "Text to speak", required=True),
                ToolParameter("voice", "string", "Voice ID"),
                ToolParameter("emotion", "string", "Emotion"),
            ]),
            # Text
            ("write_content", self._tool_write_content, "Write articles, blogs, stories", [
                ToolParameter("task", "string", "Writing task", required=True),
                ToolParameter("type", "string", "Type: article, blog, story, script, poem", default="article"),
            ]),
            ("write_code", self._tool_write_code, "Creative coding", [
                ToolParameter("task", "string", "Coding task", required=True),
            ]),
            ("write_script", self._tool_write_script, "Write scripts", [
                ToolParameter("task", "string", "Script task", required=True),
            ]),
            # Presentations
            ("create_presentation", self._tool_create_presentation, "Create presentation", [
                ToolParameter("task", "string", "Presentation topic", required=True),
                ToolParameter("slides", "integer", "Number of slides", default=10),
            ]),
            ("design_slide", self._tool_design_slide, "Design single slide", [
                ToolParameter("title", "string", "Slide title", required=True),
                ToolParameter("content", "string", "Slide content", required=True),
            ]),
            # Utilities
            ("brainstorm", self._tool_brainstorm, "Brainstorm ideas", [
                ToolParameter("task", "string", "Brainstorm topic", required=True),
            ]),
            ("prompt_enhance", self._tool_prompt_enhance, "Enhance prompts", [
                ToolParameter("prompt", "string", "Original prompt", required=True),
                ToolParameter("type", "string", "Type: image, video, text", default="image"),
            ]),
        ]

        for name, func, desc, params in tools:
            metadata = ToolMetadata(
                name=name,
                description=desc,
                category="creative",
                parameters=params,
                returns={"type": "object"},
                sandboxed=True,
                tags=["creative", "generation"],
            )
            self.tool_registry.register(name, func, metadata)

    # -------------------------------------------------------------------------
    # TASK PROCESSING
    # -------------------------------------------------------------------------

    async def process_task(
        self,
        task: str,
        context: AgentContext,
    ) -> Any:
        """Process creative task."""
        task_lower = task.lower()

        # Route to appropriate generator
        if any(kw in task_lower for kw in ["image", "picture", "photo", "illustration", "draw", "generate"]):
            if "video" in task_lower or "animate" in task_lower:
                return await self._handle_video_generation(task, context)
            return await self._handle_image_generation(task, context)

        elif any(kw in task_lower for kw in ["video", "movie", "clip", "animation"]):
            return await self._handle_video_generation(task, context)

        elif any(kw in task_lower for kw in ["music", "song", "audio", "sound", "beat"]):
            return await self._handle_audio_generation(task, context)

        elif any(kw in task_lower for kw in ["speech", "tts", "voice", "read", "narrate"]):
            return await self._handle_tts(task, context)

        elif any(kw in task_lower for kw in ["write", "article", "blog", "story", "script", "poem", "content"]):
            return await self._handle_writing(task, context)

        elif any(kw in task_lower for kw in ["presentation", "slides", "deck", "powerpoint", "keynote"]):
            return await self._handle_presentation(task, context)

        elif any(kw in task_lower for kw in ["code", "shader", "visualization", "p5", "three.js", "creative coding"]):
            return await self._handle_creative_coding(task, context)

        elif any(kw in task_lower for kw in ["brainstorm", "ideas", "concepts", "prompt"]):
            return await self._handle_brainstorm(task, context)

        # Default: use LLM to determine
        return await self._llm_route(task, context)

    async def process_task_stream(
        self,
        task: str,
        context: AgentContext,
    ) -> AsyncGenerator[str, None]:
        """Stream creative process."""
        yield f"🎨 Creative Agent: {task[:100]}...\n\n"

        task_lower = task.lower()
        if "image" in task_lower and "video" not in task_lower:
            async for chunk in self._handle_image_generation_stream(task, context):
                yield chunk
        elif "video" in task_lower or "animate" in task_lower:
            async for chunk in self._handle_video_generation_stream(task, context):
                yield chunk
        elif "music" in task_lower or "audio" in task_lower:
            async for chunk in self._handle_audio_generation_stream(task, context):
                yield chunk
        elif "write" in task_lower or "content" in task_lower:
            async for chunk in self._handle_writing_stream(task, context):
                yield chunk
        elif "presentation" in task_lower or "slides" in task_lower:
            async for chunk in self._handle_presentation_stream(task, context):
                yield chunk
        else:
            async for chunk in self._llm_route_stream(task, context):
                yield chunk

    # -------------------------------------------------------------------------
    # GENERATION HANDLERS
    # -------------------------------------------------------------------------

    async def _handle_image_generation(self, task: str, context: AgentContext) -> GenerationResult:
        """Handle image generation request."""
        prompt = await self._extract_prompt(task, "image")
        enhanced = await self._enhance_prompt(prompt, "image")
        return await self._generate_image(enhanced, context)

    async def _handle_image_generation_stream(self, task: str, context: AgentContext) -> AsyncGenerator[str, None]:
        yield "🖼️ **Image Generation**\n\n"

        prompt = await self._extract_prompt(task, "image")
        yield f"📝 Prompt: {prompt}\n\n"

        enhanced = await self._enhance_prompt(prompt, "image")
        if enhanced != prompt:
            yield f"✨ Enhanced: {enhanced}\n\n"

        yield "🎨 Generating...\n\n"

        result = await self._generate_image(enhanced, context)

        if result.success:
            for f in result.files:
                yield f"✅ Generated: `{f}`\n"
            yield f"\n⏱️ Time: {result.generation_time:.1f}s\n"
            if result.metadata.get("revised_prompt"):
                yield f"📝 Revised prompt: {result.metadata['revised_prompt']}\n"
        else:
            yield f"❌ Failed: {result.error}\n"

    async def _handle_video_generation(self, task: str, context: AgentContext) -> GenerationResult:
        prompt = await self._extract_prompt(task, "video")
        enhanced = await self._enhance_prompt(prompt, "video")
        return await self._generate_video(enhanced, context)

    async def _handle_video_generation_stream(self, task: str, context: AgentContext) -> AsyncGenerator[str, None]:
        yield "🎬 **Video Generation** (this may take 1-5 minutes)\n\n"

        prompt = await self._extract_prompt(task, "video")
        yield f"📝 Prompt: {prompt}\n\n"

        enhanced = await self._enhance_prompt(prompt, "video")
        if enhanced != prompt:
            yield f"✨ Enhanced: {enhanced}\n\n"

        yield "🎥 Generating video...\n\n"

        result = await self._generate_video(enhanced, context)

        if result.success:
            for f in result.files:
                yield f"✅ Generated: `{f}`\n"
            yield f"\n⏱️ Time: {result.generation_time:.1f}s\n"
        else:
            yield f"❌ Failed: {result.error}\n"

    async def _handle_audio_generation(self, task: str, context: AgentContext) -> GenerationResult:
        prompt = await self._extract_prompt(task, "audio")
        return await self._generate_music(prompt, context)

    async def _handle_audio_generation_stream(self, task: str, context: AgentContext) -> AsyncGenerator[str, None]:
        yield "🎵 **Music/Audio Generation**\n\n"

        prompt = await self._extract_prompt(task, "audio")
        yield f"📝 Prompt: {prompt}\n\n"

        result = await self._generate_music(prompt, context)

        if result.success:
            for f in result.files:
                yield f"✅ Generated: `{f}`\n"
        else:
            yield f"❌ Failed: {result.error}\n"

    async def _handle_tts(self, task: str, context: AgentContext) -> GenerationResult:
        text = await self._extract_prompt(task, "tts")
        return await self._text_to_speech(text, context)

    async def _handle_writing(self, task: str, context: AgentContext) -> str:
        content_type = self._detect_writing_type(task)
        return await self._write_content(task, content_type, context)

    async def _handle_writing_stream(self, task: str, context: AgentContext) -> AsyncGenerator[str, None]:
        yield "✍️ **Writing**\n\n"

        content_type = self._detect_writing_type(task)
        yield f"📝 Type: {content_type}\n\n"

        async for chunk in self._write_content_stream(task, content_type, context):
            yield chunk

    async def _handle_presentation(self, task: str, context: AgentContext) -> str:
        return await self._create_presentation(task, context)

    async def _handle_presentation_stream(self, task: str, context: AgentContext) -> AsyncGenerator[str, None]:
        yield "📊 **Creating Presentation**\n\n"

        async for chunk in self._create_presentation_stream(task, context):
            yield chunk

    async def _handle_creative_coding(self, task: str, context: AgentContext) -> str:
        return await self._write_code(task, context)

    async def _handle_brainstorm(self, task: str, context: AgentContext) -> str:
        return await self._brainstorm(task, context)

    async def _llm_route(self, task: str, context: AgentContext) -> Any:
        """Use LLM to determine creative approach."""
        if not self.llm_client:
            return "Creative task received but LLM not available for routing."

        prompt = f"""Determine the best creative approach for: "{task}"

Options: image, video, music, speech, writing, presentation, code, brainstorm

Return JSON: {{"type": "...", "reason": "...", "prompt": "extracted prompt" }}"""

        response = await self._call_llm([
            {"role": "system", "content": "Route creative tasks. Return JSON only."},
            {"role": "user", "content": prompt},
        ])

        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                routing = json.loads(json_match.group())
                return await self._execute_creative(routing["type"], routing["prompt"], context)
        except Exception:
            pass

        return "Could not determine creative approach."

    async def _llm_route_stream(self, task: str, context: AgentContext) -> AsyncGenerator[str, None]:
        yield "🤔 Determining creative approach...\n\n"
        result = await self._llm_route(task, context)
        if isinstance(result, str):
            yield result
        else:
            yield str(result)

    # -------------------------------------------------------------------------
    # GENERATION IMPLEMENTATIONS
    # -------------------------------------------------------------------------

    async def _extract_prompt(self, task: str, media_type: str) -> str:
        """Extract core prompt from task description."""
        if not self.llm_client:
            return task.replace(f"generate {media_type}", "").replace(f"create {media_type}", "").strip()

        prompt = f"""Extract the core creative prompt for {media_type} generation from this request:

Request: {task}

Return only the prompt, optimized for {media_type} generation."""

        response = await self._call_llm([
            {"role": "system", "content": f"Extract {media_type} prompts. Return only the prompt."},
            {"role": "user", "content": prompt},
        ])

        return response.strip()

    async def _enhance_prompt(self, prompt: str, media_type: str) -> str:
        """Enhance prompt for better generation."""
        if not self.llm_client:
            return prompt

        enhancement_templates = {
            "image": "Enhance this image prompt with style, lighting, composition, camera settings, and quality modifiers. Return enhanced prompt only.",
            "video": "Enhance this video prompt with motion, camera movement, style, duration hints. Return enhanced prompt only.",
            "audio": "Enhance this music/audio prompt with genre, instrumentation, mood, tempo, structure. Return enhanced prompt only.",
        }

        template = enhancement_templates.get(media_type, "Enhance this prompt. Return enhanced only.")

        response = await self._call_llm([
            {"role": "system", "content": template},
            {"role": "user", "content": prompt},
        ])

        return response.strip()

    async def _generate_image(self, prompt: str, context: AgentContext) -> GenerationResult:
        """Generate image using available providers."""
        start = time.time()

        # Try DALL-E 3 via OpenAI
        openai_key = get_settings().openai_api_key
        if openai_key:
            return await self._generate_image_dalle3(prompt, openai_key)

        # Try local Stable Diffusion (if available)
        # This would connect to local SD API

        return GenerationResult(
            success=False,
            type="image",
            error="No image generation provider configured (need OpenAI API key)",
            generation_time=time.time() - start,
        )

    async def _generate_image_dalle3(self, prompt: str, api_key: str) -> GenerationResult:
        start = time.time()

        if not self._session:
            return GenerationResult(success=False, type="image", error="No HTTP session")

        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "dall-e-3",
                "prompt": prompt,
                "n": 1,
                "size": "1024x1024",
                "quality": "hd",
                "response_format": "b64_json",
            }

            async with self._session.post(
                "https://api.openai.com/v1/images/generations",
                headers=headers,
                json=payload,
            ) as resp:
                data = await resp.json()

            if "data" in data and data["data"]:
                b64_image = data["data"][0]["b64_json"]
                revised = data["data"][0].get("revised_prompt", prompt)

                # Save
                filename = f"img_{int(time.time())}_{hashlib.md5(prompt.encode()).hexdigest()[:8]}.png"
                filepath = self._output_dir / filename

                image_data = base64.b64decode(b64_image)
                with open(filepath, "wb") as f:
                    f.write(image_data)

                return GenerationResult(
                    success=True,
                    type="image",
                    files=[str(filepath)],
                    metadata={"revised_prompt": revised, "model": "dall-e-3"},
                    generation_time=time.time() - start,
                )

            return GenerationResult(success=False, type="image", error="No image in response", generation_time=time.time() - start)

        except Exception as e:
            return GenerationResult(success=False, type="image", error=str(e), generation_time=time.time() - start)

    async def _generate_video(self, prompt: str, context: AgentContext) -> GenerationResult:
        start = time.time()

        # Would integrate with Runway, Pika, etc.
        return GenerationResult(
            success=False,
            type="video",
            error="Video generation requires API keys (Runway, Pika, etc.)",
            generation_time=time.time() - start,
        )

    async def _generate_music(self, prompt: str, context: AgentContext) -> GenerationResult:
        start = time.time()

        # Would integrate with Suno, Udio, MusicGen
        return GenerationResult(
            success=False,
            type="audio",
            error="Music generation requires API keys (Suno, Udio, MusicGen)",
            generation_time=time.time() - start,
        )

    async def _text_to_speech(self, text: str, context: AgentContext) -> GenerationResult:
        """Delegate to TTS engine via orchestrator."""
        return GenerationResult(
            success=False,
            type="speech",
            error="TTS delegation not implemented",
            generation_time=0,
        )

    async def _write_content(self, task: str, content_type: str, context: AgentContext) -> str:
        """Write content using LLM."""
        if not self.llm_client:
            return "LLM not available for writing."

        type_prompts = {
            "article": "Write a well-structured article with headings, intro, body, conclusion.",
            "blog": "Write an engaging blog post with hook, sections, actionable takeaways.",
            "story": "Write a compelling short story with characters, conflict, resolution.",
            "script": "Write a script with scene headings, dialogue, action lines.",
            "poem": "Write a poem with imagery, rhythm, emotional resonance.",
            "email": "Write a professional email with clear purpose and call to action.",
            "marketing": "Write persuasive marketing copy with hooks, benefits, CTA.",
            "technical": "Write clear technical documentation with examples.",
        }

        style = type_prompts.get(content_type, "Write high-quality content.")

        prompt = f"""{style}

Task: {task}

Requirements:
- Engaging and well-structured
- Appropriate tone for {content_type}
- Include relevant details
- Polish and refine"""

        return await self._call_llm([
            {"role": "system", "content": f"You are an expert {content_type} writer."},
            {"role": "user", "content": prompt},
        ])

    async def _write_content_stream(self, task: str, content_type: str, context: AgentContext) -> AsyncGenerator[str, None]:
        if not self.llm_client:
            yield "LLM not available for writing.\n"
            return

        type_prompts = {
            "article": "Write a well-structured article with headings, intro, body, conclusion.",
            "blog": "Write an engaging blog post with hook, sections, actionable takeaways.",
            "story": "Write a compelling short story with characters, conflict, resolution.",
            "script": "Write a script with scene headings, dialogue, action lines.",
            "poem": "Write a poem with imagery, rhythm, emotional resonance.",
            "email": "Write a professional email with clear purpose and call to action.",
            "marketing": "Write persuasive marketing copy with hooks, benefits, CTA.",
            "technical": "Write clear technical documentation with examples.",
        }

        style = type_prompts.get(content_type, "Write high-quality content.")

        prompt = f"""{style}

Task: {task}"""

        async for chunk in self._call_llm_stream([
            {"role": "system", "content": f"You are an expert {content_type} writer."},
            {"role": "user", "content": prompt},
        ]):
            yield chunk

    def _detect_writing_type(self, task: str) -> str:
        task_lower = task.lower()
        if any(kw in task_lower for kw in ["article", "blog", "post"]):
            return "article" if "article" in task_lower else "blog"
        if any(kw in task_lower for kw in ["story", "tale", "fiction", "narrative"]):
            return "story"
        if any(kw in task_lower for kw in ["script", "screenplay", "dialogue"]):
            return "script"
        if any(kw in task_lower for kw in ["poem", "poetry", "verse", "rhyme"]):
            return "poem"
        if any(kw in task_lower for kw in ["email", "mail"]):
            return "email"
        if any(kw in task_lower for kw in ["marketing", "copy", "advertisement", "sales"]):
            return "marketing"
        if any(kw in task_lower for kw in ["technical", "documentation", "docs", "readme"]):
            return "technical"
        return "article"

    async def _write_code(self, task: str, context: AgentContext) -> str:
        """Creative coding - shaders, visualizations, etc."""
        if not self.llm_client:
            return "LLM not available."

        prompt = f"""Write creative code for: {task}

Requirements:
- Complete, runnable code
- Comments explaining key parts
- Visual/creative focus
- Use appropriate libraries (p5.js, three.js, shaders, etc.)"""

        return await self._call_llm([
            {"role": "system", "content": "You are a creative coder. Write beautiful, functional code."},
            {"role": "user", "content": prompt},
        ])

    async def _create_presentation(self, task: str, context: AgentContext) -> str:
        """Create presentation outline/slides."""
        if not self.llm_client:
            return "LLM not available."

        prompt = f"""Create a presentation for: {task}

Output as JSON:
{{
  "title": "Presentation Title",
  "slides": [
    {{"title": "Slide 1", "content": "• Point 1\\n• Point 2", "notes": "Speaker notes"}},
    ...
  ]
}}

Make it professional, visually structured, with clear hierarchy."""

        response = await self._call_llm([
            {"role": "system", "content": "Create presentation outlines as JSON."},
            {"role": "user", "content": prompt},
        ])

        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                lines = [f"# {data.get('title', 'Presentation')}\n"]
                for i, slide in enumerate(data.get('slides', []), 1):
                    lines.append(f"\n## Slide {i}: {slide.get('title', '')}")
                    lines.append(slide.get('content', ''))
                    if slide.get('notes'):
                        lines.append(f"*Notes: {slide['notes']}*")
                return "\n".join(lines)
        except Exception:
            pass

        return response

    async def _create_presentation_stream(self, task: str, context: AgentContext) -> AsyncGenerator[str, None]:
        yield "📊 **Creating Presentation**\n\n"

        result = await self._create_presentation(task, context)
        yield result

    async def _brainstorm(self, task: str, context: AgentContext) -> str:
        """Brainstorm ideas."""
        if not self.llm_client:
            return "LLM not available."

        prompt = f"""Brainstorm creative ideas for: {task}

Provide:
1. 10 diverse concepts
2. For each: title, one-liner, key differentiator
3. Group by theme/category
4. Pick top 3 with rationale"""

        return await self._call_llm([
            {"role": "system", "content": "You are a creative strategist. Generate diverse, innovative ideas."},
            {"role": "user", "content": prompt},
        ])

    async def _execute_creative(self, creative_type: str, prompt: str, context: AgentContext) -> Any:
        """Execute specific creative type."""
        handlers = {
            "image": lambda: self._generate_image(prompt, context),
            "video": lambda: self._generate_video(prompt, context),
            "music": lambda: self._generate_music(prompt, context),
            "speech": lambda: self._text_to_speech(prompt, context),
            "writing": lambda: self._write_content(prompt, "article", context),
            "presentation": lambda: self._create_presentation(prompt, context),
            "code": lambda: self._write_code(prompt, context),
            "brainstorm": lambda: self._brainstorm(prompt, context),
        }

        handler = handlers.get(creative_type)
        if handler:
            return await handler()

        return f"Unknown creative type: {creative_type}"

    # -------------------------------------------------------------------------
    # TOOL IMPLEMENTATIONS
    # -------------------------------------------------------------------------

    async def _tool_generate_image(self, params: dict) -> dict:
        prompt = params.get("prompt", "")
        result = await self._generate_image(prompt, AgentContext(conversation_id="tool"))
        return result.__dict__

    async def _tool_edit_image(self, params: dict) -> dict:
        return {"success": False, "error": "Image editing requires provider API"}

    async def _tool_upscale_image(self, params: dict) -> dict:
        return {"success": False, "error": "Upscaling requires provider API"}

    async def _tool_image_variations(self, params: dict) -> dict:
        return {"success": False, "error": "Variations require provider API"}

    async def _tool_generate_video(self, params: dict) -> dict:
        prompt = params.get("prompt", "")
        result = await self._generate_video(prompt, AgentContext(conversation_id="tool"))
        return result.__dict__

    async def _tool_animate_image(self, params: dict) -> dict:
        return {"success": False, "error": "Animation requires provider API"}

    async def _tool_generate_audio(self, params: dict) -> dict:
        return {"success": False, "error": "Audio generation requires provider API"}

    async def _tool_generate_music(self, params: dict) -> dict:
        prompt = params.get("prompt", "")
        result = await self._generate_music(prompt, AgentContext(conversation_id="tool"))
        return result.__dict__

    async def _tool_text_to_speech(self, params: dict) -> dict:
        return {"success": False, "error": "Use TTS engine directly"}

    async def _tool_write_content(self, params: dict) -> dict:
        task = params.get("task", "")
        content_type = params.get("type", "article")
        result = await self._write_content(task, content_type, AgentContext(conversation_id="tool"))
        return {"content": result}

    async def _tool_write_code(self, params: dict) -> dict:
        task = params.get("task", "")
        result = await self._write_code(task, AgentContext(conversation_id="tool"))
        return {"code": result}

    async def _tool_write_script(self, params: dict) -> dict:
        task = params.get("task", "")
        result = await self._write_content(task, "script", AgentContext(conversation_id="tool"))
        return {"script": result}

    async def _tool_create_presentation(self, params: dict) -> dict:
        task = params.get("task", "")
        result = await self._create_presentation(task, AgentContext(conversation_id="tool"))
        return {"presentation": result}

    async def _tool_design_slide(self, params: dict) -> dict:
        return {"success": False, "error": "Use create_presentation"}

    async def _tool_brainstorm(self, params: dict) -> dict:
        task = params.get("task", "")
        result = await self._brainstorm(task, AgentContext(conversation_id="tool"))
        return {"ideas": result}

    async def _tool_prompt_enhance(self, params: dict) -> dict:
        prompt = params.get("prompt", "")
        media_type = params.get("type", "image")
        enhanced = await self._enhance_prompt(prompt, media_type)
        return {"original": prompt, "enhanced": enhanced}

    # -------------------------------------------------------------------------
    # EXPORTS
    # -------------------------------------------------------------------------

__all__ = [
    "CreativeAgent",
    "GenerationRequest",
    "GenerationResult",
]