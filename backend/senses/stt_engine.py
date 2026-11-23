"""
JARVIS OMEGA - Streaming Speech-to-Text Engine
Multi-provider STT with streaming support: Whisper (local), Deepgram (cloud).
Features VAD endpoint detection, interim/final results, and session management.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Optional

import numpy as np

from config import get_settings, settings

logger = logging.getLogger("jarvis.senses.stt")


# =============================================================================
# DATA MODELS
# =============================================================================

class STTProvider(str, Enum):
    WHISPER_LOCAL = "whisper_local"
    WHISPER_API = "whisper_api"
    DEEPGRAM = "deepgram"
    GOOGLE = "google"
    AZURE = "azure"


@dataclass
class TranscriptionSegment:
    """Single transcription segment."""
    text: str
    start: float
    end: float
    confidence: float
    speaker: int | None = None
    language: str | None = None
    is_final: bool = True


@dataclass
class TranscriptionResult:
    """Complete transcription result."""
    segments: list[TranscriptionSegment]
    full_text: str
    language: str | None = None
    duration: float = 0.0
    processing_time: float = 0.0
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)


@dataclass
class STTConfig:
    """STT engine configuration."""
    provider: STTProvider = STTProvider.WHISPER_LOCAL
    model: str = "large-v3"
    language: str | None = None  # None = auto-detect
    translate: bool = False
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 100

    # Local Whisper (faster-whisper)
    whisper_local: dict[str, Any] = field(default_factory=lambda: {
        "device": "cuda",
        "compute_type": "float16",
        "beam_size": 5,
        "vad_filter": True,
        "vad_parameters": {
            "threshold": 0.5,
            "min_speech_duration_ms": 250,
            "max_speech_duration_s": 30,
            "min_silence_duration_ms": 100,
            "speech_pad_ms": 300,
        },
    })

    # Deepgram Streaming
    deepgram: dict[str, Any] = field(default_factory=lambda: {
        "model": "nova-2",
        "language": "en",
        "smart_format": True,
        "punctuate": True,
        "diarize": True,
        "interim_results": True,
        "endpointing": 300,
        "utterance_end_ms": 1000,
    })

    # VAD settings
    vad: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "max_speech_duration_s": 30.0,
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 300,
    })


# =============================================================================
# BASE PROVIDER INTERFACE
# =============================================================================

class STTProviderBase(ABC):
    """Abstract base class for STT providers."""

    def __init__(self, config: STTConfig):
        self.config = config
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the provider."""
        pass

    @abstractmethod
    async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe complete audio buffer."""
        pass

    @abstractmethod
    async def stream_transcribe(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
    ) -> AsyncGenerator[TranscriptionSegment, None]:
        """Stream transcription from audio generator."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Cleanup resources."""
        pass

    @property
    def is_ready(self) -> bool:
        return self._initialized


# =============================================================================
# LOCAL WHISPER PROVIDER (faster-whisper)
# =============================================================================

class WhisperLocalProvider(STTProviderBase):
    """Local Whisper transcription using faster-whisper (CTranslate2)."""

    def __init__(self, config: STTConfig):
        super().__init__(config)
        self._model: Any = None
        self._vad_model: Any = None

    async def initialize(self) -> bool:
        try:
            from faster_whisper import WhisperModel

            model_size = self.config.model
            device = self.config.whisper_local.get("device", "cuda")
            compute_type = self.config.whisper_local.get("compute_type", "float16")

            logger.info(f"Loading Whisper model: {model_size} on {device} ({compute_type})")

            self._model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=str(Path.home() / ".cache" / "whisper"),
            )

            # Load VAD if enabled
            if self.config.vad.get("enabled", True):
                await self._load_vad()

            self._initialized = True
            logger.info("WhisperLocalProvider initialized")
            return True

        except ImportError:
            logger.error("faster-whisper not installed. Install: pip install faster-whisper")
            return False
        except Exception as e:
            logger.exception(f"Failed to initialize WhisperLocalProvider: {e}")
            return False

    async def _load_vad(self) -> None:
        """Load Silero VAD for endpoint detection."""
        try:
            import torch
            self._vad_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            logger.info("Silero VAD loaded for Whisper")
        except Exception as e:
            logger.warning(f"Could not load VAD: {e}")

    def _run_vad(self, audio_chunk: np.ndarray) -> bool:
        """Run VAD on audio chunk."""
        if self._vad_model is None:
            return True

        try:
            import torch

            if audio_chunk.dtype == np.int16:
                audio_tensor = torch.from_numpy(audio_chunk.astype(np.float32) / 32768.0)
            else:
                audio_tensor = torch.from_numpy(audio_chunk.astype(np.float32))

            # Silero expects 512 samples at 16kHz
            if len(audio_tensor) != 512:
                if len(audio_tensor) > 512:
                    audio_tensor = audio_tensor[:512]
                else:
                    padding = torch.zeros(512 - len(audio_tensor))
                    audio_tensor = torch.cat([audio_tensor, padding])

            with torch.no_grad():
                speech_prob = self._vad_model(audio_tensor, self.config.sample_rate).item()

            return speech_prob > self.config.vad.get("threshold", 0.5)
        except Exception:
            return True

    async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe complete audio buffer."""
        if not self._initialized or self._model is None:
            raise RuntimeError("Provider not initialized")

        start_time = time.time()

        # Ensure float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        # Run transcription in thread pool
        loop = asyncio.get_event_loop()
        segments, info = await loop.run_in_executor(
            None,
            lambda: self._model.transcribe(
                audio,
                language=self.config.language,
                task="translate" if self.config.translate else "transcribe",
                beam_size=self.config.whisper_local.get("beam_size", 5),
                vad_filter=self.config.whisper_local.get("vad_filter", True),
                vad_parameters=self.config.whisper_local.get("vad_parameters", {}),
            ),
        )

        # Convert segments
        transcription_segments = []
        full_text_parts = []

        for seg in segments:
            transcription_segments.append(TranscriptionSegment(
                text=seg.text.strip(),
                start=seg.start,
                end=seg.end,
                confidence=getattr(seg, 'avg_logprob', 1.0),
                language=info.language,
                is_final=True,
            ))
            full_text_parts.append(seg.text.strip())

        processing_time = time.time() - start_time

        return TranscriptionResult(
            segments=transcription_segments,
            full_text=" ".join(full_text_parts),
            language=info.language,
            duration=len(audio) / self.config.sample_rate,
            processing_time=processing_time,
        )

    async def stream_transcribe(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
    ) -> AsyncGenerator[TranscriptionSegment, None]:
        """
        Stream transcription using sliding window approach.
        Note: faster-whisper doesn't natively support streaming,
        so we accumulate and transcribe windows.
        """
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

        # Buffer for accumulating audio
        buffer = np.array([], dtype=np.float32)
        window_samples = int(self.config.sample_rate * 5)  # 5 second window
        step_samples = int(self.config.sample_rate * 1)    # 1 second step
        last_transcribed_end = 0.0

        async for chunk in audio_stream:
            # Convert to float32
            if chunk.dtype == np.int16:
                chunk = chunk.astype(np.float32) / 32768.0

            buffer = np.concatenate([buffer, chunk])

            # Process when we have enough audio
            while len(buffer) >= window_samples:
                window = buffer[:window_samples]
                buffer = buffer[step_samples:]

                # Run VAD on window
                if self._vad_model is not None:
                    # Quick VAD check on first chunk of window
                    if not self._run_vad(window[:512]):
                        continue

                # Transcribe window
                try:
                    loop = asyncio.get_event_loop()
                    segments, info = await loop.run_in_executor(
                        None,
                        lambda: self._model.transcribe(
                            window,
                            language=self.config.language,
                            task="translate" if self.config.translate else "transcribe",
                            beam_size=1,  # Faster for streaming
                            vad_filter=True,
                        ),
                    )

                    for seg in segments:
                        # Adjust timestamps
                        adjusted_start = seg.start + last_transcribed_end
                        adjusted_end = seg.end + last_transcribed_end

                        yield TranscriptionSegment(
                            text=seg.text.strip(),
                            start=adjusted_start,
                            end=adjusted_end,
                            confidence=getattr(seg, 'avg_logprob', 1.0),
                            language=info.language,
                            is_final=False,  # Interim
                        )

                    last_transcribed_end += step_samples / self.config.sample_rate

                except Exception as e:
                    logger.error(f"Streaming transcription error: {e}")

    async def shutdown(self) -> None:
        self._model = None
        self._vad_model = None
        self._initialized = False


# =============================================================================
# DEEPGRAM PROVIDER (Cloud Streaming)
# =============================================================================

class DeepgramProvider(STTProviderBase):
    """Deepgram streaming STT provider."""

    def __init__(self, config: STTConfig):
        super().__init__(config)
        self._client: Any = None
        self._connection: Any = None

    async def initialize(self) -> bool:
        try:
            from deepgram import DeepgramClient, DeepgramClientOptions

            api_key = get_settings().deepgram_api_key
            if not api_key:
                logger.error("Deepgram API key not configured")
                return False

            config_options = DeepgramClientOptions(
                api_key=api_key,
                options={"keepalive": "true"},
            )
            self._client = DeepgramClient("", config_options)

            self._initialized = True
            logger.info("DeepgramProvider initialized")
            return True

        except ImportError:
            logger.error("deepgram-sdk not installed. Install: pip install deepgram-sdk")
            return False
        except Exception as e:
            logger.exception(f"Failed to initialize DeepgramProvider: {e}")
            return False

    async def _create_connection(self):
        """Create Deepgram live transcription connection."""
        if not self._client:
            return None

        options = self.config.deepgram
        return self._client.listen.asyncwebsocket.v("1")

    async def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """Transcribe complete audio (non-streaming)."""
        # For non-streaming, use Deepgram REST API
        import httpx

        api_key = get_settings().deepgram_api_key
        if not api_key:
            raise RuntimeError("Deepgram API key not configured")

        # Convert audio to bytes
        if audio.dtype == np.float32:
            audio = (audio * 32767).astype(np.int16)
        audio_bytes = audio.tobytes()

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.deepgram.com/v1/listen",
                headers={
                    "Authorization": f"Token {api_key}",
                    "Content-Type": "audio/wav",
                },
                params={
                    "model": self.config.deepgram.get("model", "nova-2"),
                    "language": self.config.deepgram.get("language", "en"),
                    "smart_format": str(self.config.deepgram.get("smart_format", True)).lower(),
                    "punctuate": str(self.config.deepgram.get("punctuate", True)).lower(),
                    "diarize": str(self.config.deepgram.get("diarize", True)).lower(),
                },
                content=audio_bytes,
            )
            response.raise_for_status()
            data = response.json()

        # Parse response
        results = data.get("results", {})
        channels = results.get("channels", [])
        if not channels:
            return TranscriptionResult(segments=[], full_text="")

        alternatives = channels[0].get("alternatives", [])
        if not alternatives:
            return TranscriptionResult(segments=[], full_text="")

        transcript = alternatives[0].get("transcript", "")
        words = alternatives[0].get("words", [])

        segments = []
        for word in words:
            segments.append(TranscriptionSegment(
                text=word.get("word", ""),
                start=word.get("start", 0),
                end=word.get("end", 0),
                confidence=word.get("confidence", 1.0),
                speaker=word.get("speaker"),
            ))

        return TranscriptionResult(
            segments=segments,
            full_text=transcript,
            language=self.config.deepgram.get("language"),
        )

    async def stream_transcribe(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
    ) -> AsyncGenerator[TranscriptionSegment, None]:
        """Stream transcription via Deepgram WebSocket."""
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

        # This would implement the full Deepgram WebSocket protocol
        # For brevity, showing structure - full implementation needs
        # proper WebSocket message handling with callbacks
        logger.warning("Deepgram streaming not fully implemented - use transcribe() for now")

        async for chunk in audio_stream:
            # Send to Deepgram WebSocket
            # Receive interim/final results
            # Yield TranscriptionSegment
            pass

    async def shutdown(self) -> None:
        if self._connection:
            await self._connection.finish()
        self._client = None
        self._initialized = False


# =============================================================================
# STT ENGINE (Orchestrator)
# =============================================================================

class STTEngine:
    """
    High-level STT engine managing provider selection,
    audio preprocessing, and result post-processing.
    """

    def __init__(
        self,
        websocket_manager: Any = None,
        event_bus: Any = None,
        config: STTConfig | None = None,
    ):
        self.websocket_manager = websocket_manager
        self.event_bus = event_bus
        self.config = config or STTConfig()

        self._provider: STTProviderBase | None = None
        self._running = False
        self._stream_sessions: dict[str, "StreamSession"] = {}

        # Stats
        self._stats = {
            "total_requests": 0,
            "total_audio_seconds": 0.0,
            "avg_latency_ms": 0.0,
            "errors": 0,
        }

        logger.info("STTEngine initialized")

    async def start(self) -> bool:
        """Initialize and start the STT engine."""
        if self._running:
            return True

        # Create provider based on config
        self._provider = self._create_provider()
        if not self._provider:
            return False

        success = await self._provider.initialize()
        if not success:
            logger.error("Failed to initialize STT provider")
            return False

        self._running = True
        logger.info(f"STTEngine started with provider: {self.config.provider.value}")
        return True

    async def stop(self) -> None:
        """Stop the STT engine."""
        self._running = False

        # Close all stream sessions
        for session in self._stream_sessions.values():
            await session.close()

        if self._provider:
            await self._provider.shutdown()

        logger.info("STTEngine stopped")

    def _create_provider(self) -> STTProviderBase | None:
        """Factory method to create provider."""
        provider_map = {
            STTProvider.WHISPER_LOCAL: WhisperLocalProvider,
            STTProvider.WHISPER_API: WhisperLocalProvider,  # Fallback to local
            STTProvider.DEEPGRAM: DeepgramProvider,
        }

        provider_class = provider_map.get(self.config.provider)
        if provider_class:
            return provider_class(self.config)

        logger.error(f"Unknown STT provider: {self.config.provider}")
        return None

    async def transcribe(self, audio: np.ndarray | bytes) -> TranscriptionResult:
        """
        Transcribe complete audio buffer.

        Args:
            audio: Audio data as numpy array (float32 [-1,1] or int16) or bytes
        """
        if not self._running or not self._provider:
            raise RuntimeError("STT engine not running")

        start_time = time.time()

        # Convert to numpy
        if isinstance(audio, bytes):
            audio = np.frombuffer(audio, dtype=np.int16)

        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        # Update stats
        duration = len(audio) / self.config.sample_rate
        self._stats["total_requests"] += 1
        self._stats["total_audio_seconds"] += duration

        # Transcribe
        try:
            result = await self._provider.transcribe(audio)
            latency = (time.time() - start_time) * 1000

            # Update rolling average latency
            n = self._stats["total_requests"]
            self._stats["avg_latency_ms"] = (
                (self._stats["avg_latency_ms"] * (n - 1) + latency) / n
            )

            # Emit event
            if self.event_bus:
                await self.event_bus.emit("stt.transcription", {
                    "text": result.full_text,
                    "language": result.language,
                    "duration": result.duration,
                    "latency_ms": latency,
                })

            return result

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Transcription error: {e}")
            raise

    async def start_stream_session(
        self,
        session_id: str,
        on_segment: Callable[[TranscriptionSegment], Any],
        config: STTConfig | None = None,
    ) -> "StreamSession":
        """Start a streaming transcription session."""
        if not self._running or not self._provider:
            raise RuntimeError("STT engine not running")

        session = StreamSession(
            session_id=session_id,
            engine=self,
            provider=self._provider,
            on_segment=on_segment,
            config=config or self.config,
        )

        self._stream_sessions[session_id] = session
        await session.start()

        return session

    async def stop_stream_session(self, session_id: str) -> TranscriptionResult | None:
        """Stop a streaming session and return final result."""
        session = self._stream_sessions.pop(session_id, None)
        if session:
            return await session.stop()
        return None

    def feed_stream_audio(self, session_id: str, audio: np.ndarray | bytes) -> bool:
        """Feed audio to a streaming session."""
        session = self._stream_sessions.get(session_id)
        if not session:
            return False

        if isinstance(audio, bytes):
            audio = np.frombuffer(audio, dtype=np.int16)

        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        session.feed_audio(audio)
        return True

    def get_stats(self) -> dict[str, Any]:
        return self._stats.copy()

    async def switch_provider(self, provider: STTProvider) -> bool:
        """Switch STT provider at runtime."""
        if not self._running:
            self.config.provider = provider
            return await self.start()

        # Stop current
        await self.stop()

        # Switch and restart
        self.config.provider = provider
        return await self.start()


# =============================================================================
# STREAM SESSION
# =============================================================================

class StreamSession:
    """Manages a single streaming transcription session."""

    def __init__(
        self,
        session_id: str,
        engine: STTEngine,
        provider: STTProviderBase,
        on_segment: Callable[[TranscriptionSegment], Any],
        config: STTConfig,
    ):
        self.session_id = session_id
        self.engine = engine
        self.provider = provider
        self.on_segment = on_segment
        self.config = config

        self._audio_queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._process_task: asyncio.Task | None = None
        self._running = False
        self._segments: list[TranscriptionSegment] = []
        self._start_time = time.time()

    async def start(self) -> None:
        """Start the streaming session."""
        self._running = True
        self._process_task = asyncio.create_task(self._process_stream())
        logger.debug(f"Stream session started: {self.session_id}")

    async def stop(self) -> TranscriptionResult:
        """Stop session and return final result."""
        self._running = False

        if self._process_task:
            self._process_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._process_task

        # Combine all segments
        full_text = " ".join(s.text for s in self._segments if s.text)

        return TranscriptionResult(
            segments=self._segments,
            full_text=full_text,
            language=self._segments[0].language if self._segments else None,
            duration=time.time() - self._start_time,
        )

    def feed_audio(self, audio: np.ndarray) -> None:
        """Feed audio chunk to session."""
        try:
            self._audio_queue.put_nowait(audio)
        except asyncio.QueueFull:
            logger.warning(f"Stream session {self.session_id} queue full")

    async def _process_stream(self) -> None:
        """Process audio stream."""
        async def audio_generator():
            while self._running:
                try:
                    chunk = await asyncio.wait_for(self._audio_queue.get(), timeout=0.1)
                    yield chunk
                except asyncio.TimeoutError:
                    continue

        try:
            async for segment in self.provider.stream_transcribe(audio_generator()):
                self._segments.append(segment)

                # Call callback
                try:
                    await self.on_segment(segment)
                except Exception as e:
                    logger.error(f"Segment callback error: {e}")

                # Broadcast interim result via WebSocket
                if self.engine.websocket_manager:
                    await self.engine.websocket_manager.broadcast({
                        "type": "stt_interim",
                        "payload": {
                            "session_id": self.session_id,
                            "text": segment.text,
                            "start": segment.start,
                            "end": segment.end,
                            "is_final": segment.is_final,
                            "timestamp": time.time(),
                        },
                    }, room="hud")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Stream processing error: {e}")

    async def close(self) -> None:
        await self.stop()


# =============================================================================
# STREAMING TRANSCRIBER (High-level helper)
# =============================================================================

class StreamingTranscriber:
    """
    High-level streaming transcriber for simple use cases.
    Handles audio capture, VAD, and transcription automatically.
    """

    def __init__(
        self,
        stt_engine: STTEngine,
        vad_threshold: float = 0.5,
        silence_timeout: float = 2.0,
        max_utterance_duration: float = 30.0,
    ):
        self.stt_engine = stt_engine
        self.vad_threshold = vad_threshold
        self.silence_timeout = silence_timeout
        self.max_utterance_duration = max_utterance_duration

        self._session_id: str | None = None
        self._current_utterance: list[np.ndarray] = []
        self._last_speech_time: float = 0
        self._utterance_start_time: float = 0
        self._vad_model: Any = None
        self._callback: Callable[[TranscriptionResult], Any] | None = None

    async def start(self, callback: Callable[[TranscriptionResult], Any]) -> None:
        """Start listening for speech."""
        self._callback = callback
        await self._init_vad()

        self._session_id = str(uuid.uuid4())
        await self.stt_engine.start_stream_session(
            self._session_id,
            self._on_segment,
        )

        logger.info("StreamingTranscriber started")

    async def stop(self) -> None:
        """Stop listening."""
        if self._session_id:
            await self.stt_engine.stop_stream_session(self._session_id)
            self._session_id = None

    async def _init_vad(self) -> None:
        """Initialize VAD."""
        try:
            import torch
            self._vad_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
        except Exception as e:
            logger.warning(f"VAD not available: {e}")

    def _run_vad(self, audio: np.ndarray) -> bool:
        """Check if audio contains speech."""
        if self._vad_model is None:
            return True

        try:
            import torch
            if audio.dtype == np.int16:
                audio_tensor = torch.from_numpy(audio.astype(np.float32) / 32768.0)
            else:
                audio_tensor = torch.from_numpy(audio.astype(np.float32))

            if len(audio_tensor) != 512:
                if len(audio_tensor) > 512:
                    audio_tensor = audio_tensor[:512]
                else:
                    padding = torch.zeros(512 - len(audio_tensor))
                    audio_tensor = torch.cat([audio_tensor, padding])

            with torch.no_grad():
                return self._vad_model(audio_tensor, 16000).item() > self.vad_threshold
        except Exception:
            return True

    def feed_audio(self, audio: np.ndarray | bytes) -> None:
        """Feed audio for VAD + transcription."""
        if isinstance(audio, bytes):
            audio = np.frombuffer(audio, dtype=np.int16)

        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        is_speech = self._run_vad(audio)
        now = time.time()

        if is_speech:
            self._last_speech_time = now
            if not self._current_utterance:
                self._utterance_start_time = now
            self._current_utterance.append(audio)

            # Check max duration
            if now - self._utterance_start_time > self.max_utterance_duration:
                self._finalize_utterance()

        elif self._current_utterance:
            # Check silence timeout
            if now - self._last_speech_time > self.silence_timeout:
                self._finalize_utterance()

        # Feed to STT engine
        if self._session_id:
            self.stt_engine.feed_stream_audio(self._session_id, audio)

    def _finalize_utterance(self) -> None:
        """Finalize current utterance."""
        if self._current_utterance:
            utterance_audio = np.concatenate(self._current_utterance)
            self._current_utterance = []

            logger.debug(f"Utterance finalized: {len(utterance_audio)} samples")

        # The STT engine will process and call _on_segment

    async def _on_segment(self, segment: TranscriptionSegment) -> None:
        """Handle transcription segment."""
        if segment.is_final and self._callback:
            result = TranscriptionResult(
                segments=[segment],
                full_text=segment.text,
                language=segment.language,
            )
            await self._callback(result)


# =============================================================================
# FACTORY
# =============================================================================

async def create_stt_engine(
    websocket_manager: Any = None,
    event_bus: Any = None,
    provider: str | STTProvider | None = None,
) -> STTEngine:
    """Factory to create and start STT engine."""
    config = STTConfig()
    if provider:
        config.provider = STTProvider(provider) if isinstance(provider, str) else provider

    engine = STTEngine(
        websocket_manager=websocket_manager,
        event_bus=event_bus,
        config=config,
    )
    await engine.start()
    return engine


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "STTEngine",
    "StreamingTranscriber",
    "STTProvider",
    "STTConfig",
    "TranscriptionSegment",
    "TranscriptionResult",
    "StreamSession",
    "create_stt_engine",
    "WhisperLocalProvider",
    "DeepgramProvider",
    "STTProviderBase",
]