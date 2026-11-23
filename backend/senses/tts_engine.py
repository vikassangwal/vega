"""
JARVIS OMEGA - Neural Text-to-Speech Engine
Multi-provider TTS with emotion control, voice cloning, streaming:
- Edge-TTS (Free, Microsoft, high quality)
- ElevenLabs (Premium, best quality, voice cloning)
- Coqui XTTS v2 (Local, voice cloning, multilingual)
- Piper (Fast local, lightweight)
- OpenAI TTS (Cloud)
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import logging
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import numpy as np

from config import get_settings, settings

logger = logging.getLogger("jarvis.senses.tts")


# =============================================================================
# DATA MODELS
# =============================================================================

class TTSProvider(str, Enum):
    EDGE_TTS = "edge_tts"
    ELEVENLABS = "elevenlabs"
    COQUI_XTTS = "coqui_xtts"
    PIPER = "piper"
    OPENAI_TTS = "openai_tts"
    GOOGLE_TTS = "google_tts"


@dataclass
class VoiceProfile:
    """Voice configuration profile."""
    provider: TTSProvider
    voice_id: str
    name: str
    language: str = "en"
    gender: str | None = None
    description: str = ""
    # Cloning
    supports_cloning: bool = False
    clone_reference: str | None = None
    # Emotion/Style
    supports_emotion: bool = False
    default_emotion: str = "neutral"
    available_emotions: list[str] = field(default_factory=list)
    # Quality
    sample_rate: int = 22050
    bitrate: str = "64k"


@dataclass
class TTSRequest:
    """TTS synthesis request."""
    text: str
    voice_profile: VoiceProfile | None = None
    # Override voice settings
    voice_id: str | None = None
    language: str | None = None
    speed: float = 1.0
    pitch: float = 0.0
    volume: float = 1.0
    emotion: str | None = None
    emotion_strength: float = 1.0
    # Streaming
    stream: bool = True
    chunk_size: int = 1024
    # Output format
    format: str = "wav"  # wav, mp3, ogg, pcm
    sample_rate: int = 22050
    # Cloning
    clone_speaker_wav: str | None = None
    # Request metadata
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)


@dataclass
class TTSResult:
    """TTS synthesis result."""
    request_id: str
    audio_data: bytes
    format: str
    sample_rate: int
    duration: float
    processing_time: float
    voice_used: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class TTSConfig:
    """TTS engine configuration."""
    default_provider: TTSProvider = TTSProvider.EDGE_TTS
    default_voice: str = "en-US-AriaNeural"
    default_language: str = "en"
    default_speed: float = 1.0
    default_pitch: float = 0.0
    default_volume: float = 1.0

    # Provider configs
    edge_tts: dict[str, Any] = field(default_factory=lambda: {
        "rate": "+0%",
        "volume": "+0%",
        "pitch": "+0Hz",
    })

    elevenlabs: dict[str, Any] = field(default_factory=lambda: {
        "model_id": "eleven_multilingual_v2",
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.0,
        "use_speaker_boost": True,
    })

    coqui_xtts: dict[str, Any] = field(default_factory=lambda: {
        "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
        "speaker_wav": None,
        "language": "en",
        "temperature": 0.7,
        "length_penalty": 1.0,
        "repetition_penalty": 1.0,
        "top_k": 50,
        "top_p": 0.85,
    })

    piper: dict[str, Any] = field(default_factory=lambda: {
        "model_path": str(Path.home() / ".cache" / "piper" / "en_US-amy-medium.onnx"),
        "config_path": str(Path.home() / ".cache" / "piper" / "en_US-amy-medium.onnx.json"),
        "length_scale": 1.0,
        "noise_scale": 0.667,
        "noise_w_scale": 0.8,
    })

    openai_tts: dict[str, Any] = field(default_factory=lambda: {
        "model": "tts-1-hd",
        "voice": "alloy",
    })

    # Cache settings
    cache_enabled: bool = True
    cache_dir: str = str(Path.home() / ".cache" / "vega" / "tts")
    cache_max_size_mb: int = 500

    # Streaming
    stream_chunk_size: int = 1024
    stream_buffer_chunks: int = 4


# =============================================================================
# VOICE REGISTRY
# =============================================================================

# Predefined voice profiles
VOICE_REGISTRY: dict[str, VoiceProfile] = {
    # Edge-TTS voices (free, high quality)
    "en-US-AriaNeural": VoiceProfile(
        provider=TTSProvider.EDGE_TTS,
        voice_id="en-US-AriaNeural",
        name="Aria (US)",
        language="en-US",
        gender="female",
        description="Natural, conversational US English female",
        supports_emotion=True,
        available_emotions=["neutral", "cheerful", "sad", "angry", "fearful", "whispering"],
        sample_rate=24000,
    ),
    "en-US-GuyNeural": VoiceProfile(
        provider=TTSProvider.EDGE_TTS,
        voice_id="en-US-GuyNeural",
        name="Guy (US)",
        language="en-US",
        gender="male",
        description="Natural, conversational US English male",
        supports_emotion=True,
        available_emotions=["neutral", "cheerful", "sad", "angry", "fearful"],
        sample_rate=24000,
    ),
    "en-GB-SoniaNeural": VoiceProfile(
        provider=TTSProvider.EDGE_TTS,
        voice_id="en-GB-SoniaNeural",
        name="Sonia (UK)",
        language="en-GB",
        gender="female",
        description="British English female",
        supports_emotion=True,
        available_emotions=["neutral", "cheerful", "sad", "angry"],
        sample_rate=24000,
    ),
    "hi-IN-SwaraNeural": VoiceProfile(
        provider=TTSProvider.EDGE_TTS,
        voice_id="hi-IN-SwaraNeural",
        name="Swara (Hindi)",
        language="hi-IN",
        gender="female",
        description="Hindi female voice",
        supports_emotion=False,
        sample_rate=24000,
    ),
    "ja-JP-NanamiNeural": VoiceProfile(
        provider=TTSProvider.EDGE_TTS,
        voice_id="ja-JP-NanamiNeural",
        name="Nanami (Japanese)",
        language="ja-JP",
        gender="female",
        description="Japanese female voice",
        supports_emotion=False,
        sample_rate=24000,
    ),

    # ElevenLabs voices (premium)
    "eleven_rachel": VoiceProfile(
        provider=TTSProvider.ELEVENLABS,
        voice_id="21m00Tcm4TlvDq8ikWAM",
        name="Rachel",
        language="en",
        gender="female",
        description="Professional, calm female",
        supports_cloning=True,
        supports_emotion=True,
        available_emotions=["neutral", "happy", "sad", "angry", "whisper"],
        sample_rate=22050,
    ),
    "eleven_adam": VoiceProfile(
        provider=TTSProvider.ELEVENLABS,
        voice_id="pNInz6obpgDQGcFmaJgB",
        name="Adam",
        language="en",
        gender="male",
        description="Deep, authoritative male",
        supports_cloning=True,
        supports_emotion=True,
        available_emotions=["neutral", "happy", "sad", "angry"],
        sample_rate=22050,
    ),

    # Coqui XTTS (cloning)
    "xtts_clone": VoiceProfile(
        provider=TTSProvider.COQUI_XTTS,
        voice_id="xtts_clone",
        name="XTTS Clone",
        language="multi",
        description="Voice cloning with XTTS v2",
        supports_cloning=True,
        supports_emotion=False,
        sample_rate=22050,
    ),

    # Piper voices (fast local)
    "piper_amy": VoiceProfile(
        provider=TTSProvider.PIPER,
        voice_id="en_US-amy-medium",
        name="Amy (Piper)",
        language="en-US",
        gender="female",
        description="Fast local female voice",
        supports_emotion=False,
        sample_rate=22050,
    ),
    "piper_ryan": VoiceProfile(
        provider=TTSProvider.PIPER,
        voice_id="en_US-ryan-medium",
        name="Ryan (Piper)",
        language="en-US",
        gender="male",
        description="Fast local male voice",
        supports_emotion=False,
        sample_rate=22050,
    ),
}


# =============================================================================
# BASE PROVIDER INTERFACE
# =============================================================================

class TTSProviderBase(ABC):
    """Abstract base class for TTS providers."""

    def __init__(self, config: TTSConfig):
        self.config = config
        self._initialized = False
        self._voice_cache: dict[str, bytes] = {}

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the provider."""
        pass

    @abstractmethod
    async def synthesize(self, request: TTSRequest) -> TTSResult:
        """Synthesize speech from text."""
        pass

    @abstractmethod
    async def stream_synthesize(
        self,
        request: TTSRequest,
    ) -> AsyncGenerator[bytes, None]:
        """Stream synthesis chunks."""
        pass

    @abstractmethod
    async def get_available_voices(self) -> list[VoiceProfile]:
        """Get list of available voices."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Cleanup resources."""
        pass

    def _get_cache_key(self, request: TTSRequest) -> str:
        """Generate cache key for request."""
        key_data = f"{request.text}|{request.voice_id}|{request.speed}|{request.pitch}|{request.emotion}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cached(self, key: str) -> bytes | None:
        """Get cached audio."""
        if not self.config.cache_enabled:
            return None
        return self._voice_cache.get(key)

    def _set_cache(self, key: str, audio: bytes) -> None:
        """Cache audio."""
        if not self.config.cache_enabled:
            return
        # Simple LRU - remove oldest if too large
        if len(self._voice_cache) > 1000:
            oldest = next(iter(self._voice_cache))
            del self._voice_cache[oldest]
        self._voice_cache[key] = audio


# =============================================================================
# EDGE-TTS PROVIDER (Free, Microsoft)
# =============================================================================

class EdgeTTSProvider(TTSProviderBase):
    """Microsoft Edge-TTS provider - free, high quality neural voices."""

    def __init__(self, config: TTSConfig):
        super().__init__(config)
        self._edge_tts = None

    async def initialize(self) -> bool:
        try:
            import edge_tts
            self._edge_tts = edge_tts
            self._initialized = True
            logger.info("EdgeTTSProvider initialized")
            return True
        except ImportError:
            logger.error("edge-tts not installed. Install: pip install edge-tts")
            return False
        except Exception as e:
            logger.exception(f"EdgeTTS init error: {e}")
            return False

    async def synthesize(self, request: TTSRequest) -> TTSResult:
        start_time = time.time()
        voice_id = request.voice_id or request.voice_profile.voice_id if request.voice_profile else self.config.default_voice

        # Check cache
        cache_key = self._get_cache_key(request)
        cached = self._get_cached(cache_key)
        if cached:
            return TTSResult(
                request_id=request.request_id,
                audio_data=cached,
                format=request.format,
                sample_rate=request.sample_rate,
                duration=len(cached) / (request.sample_rate * 2),  # Rough estimate
                processing_time=time.time() - start_time,
                voice_used=voice_id,
            )

        # Build edge-tts communicate
        communicate = self._edge_tts.Communicate(
            text=request.text,
            voice=voice_id,
            rate=self.config.edge_tts.get("rate", "+0%"),
            volume=self.config.edge_tts.get("volume", "+0%"),
            pitch=self.config.edge_tts.get("pitch", "+0Hz"),
        )

        # Synthesize to bytes
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])

        audio_data = b"".join(audio_chunks)

        # Convert format if needed
        if request.format != "mp3":
            audio_data = await self._convert_audio(audio_data, "mp3", request.format, request.sample_rate)

        # Cache
        self._set_cache(cache_key, audio_data)

        return TTSResult(
            request_id=request.request_id,
            audio_data=audio_data,
            format=request.format,
            sample_rate=request.sample_rate,
            duration=len(audio_data) / (request.sample_rate * 2),
            processing_time=time.time() - start_time,
            voice_used=voice_id,
        )

    async def stream_synthesize(
        self,
        request: TTSRequest,
    ) -> AsyncGenerator[bytes, None]:
        voice_id = request.voice_id or request.voice_profile.voice_id if request.voice_profile else self.config.default_voice

        communicate = self._edge_tts.Communicate(
            text=request.text,
            voice=voice_id,
            rate=self.config.edge_tts.get("rate", "+0%"),
            volume=self.config.edge_tts.get("volume", "+0%"),
            pitch=self.config.edge_tts.get("pitch", "+0Hz"),
        )

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    async def get_available_voices(self) -> list[VoiceProfile]:
        voices = []
        for voice_id, profile in VOICE_REGISTRY.items():
            if profile.provider == TTSProvider.EDGE_TTS:
                voices.append(profile)
        return voices

    async def shutdown(self) -> None:
        self._initialized = False

    async def _convert_audio(self, data: bytes, from_fmt: str, to_fmt: str, sample_rate: int) -> bytes:
        """Convert audio format using ffmpeg."""
        try:
            import subprocess

            with tempfile.NamedTemporaryFile(suffix=f".{from_fmt}", delete=False) as infile:
                infile.write(data)
                in_path = infile.name

            with tempfile.NamedTemporaryFile(suffix=f".{to_fmt}", delete=False) as outfile:
                out_path = outfile.name

            cmd = [
                "ffmpeg", "-y", "-i", in_path,
                "-ar", str(sample_rate),
                "-ac", "1",
                "-f", to_fmt,
                out_path,
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()

            with open(out_path, "rb") as f:
                result = f.read()

            # Cleanup
            Path(in_path).unlink(missing_ok=True)
            Path(out_path).unlink(missing_ok=True)

            return result
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return data  # Return original


# =============================================================================
# ELEVENLABS PROVIDER (Premium)
# =============================================================================

class ElevenLabsProvider(TTSProviderBase):
    """ElevenLabs TTS provider - premium quality with voice cloning."""

    def __init__(self, config: TTSConfig):
        super().__init__(config)
        self._client: Any = None
        self._api_key: str | None = None

    async def initialize(self) -> bool:
        try:
            from elevenlabs import ElevenLabs
            from config import get_settings

            self._api_key = get_settings().elevenlabs_api_key
            if not self._api_key:
                logger.warning("ElevenLabs API key not configured")
                return False

            self._client = ElevenLabs(api_key=self._api_key)
            self._initialized = True
            logger.info("ElevenLabsProvider initialized")
            return True
        except ImportError:
            logger.error("elevenlabs not installed. Install: pip install elevenlabs")
            return False
        except Exception as e:
            logger.exception(f"ElevenLabs init error: {e}")
            return False

    async def synthesize(self, request: TTSRequest) -> TTSResult:
        if not self._initialized or not self._client:
            raise RuntimeError("ElevenLabs not initialized")

        start_time = time.time()
        voice_id = request.voice_id or request.voice_profile.voice_id if request.voice_profile else "21m00Tcm4TlvDq8ikWAM"

        # Check cache
        cache_key = self._get_cache_key(request)
        cached = self._get_cached(cache_key)
        if cached:
            return TTSResult(
                request_id=request.request_id,
                audio_data=cached,
                format=request.format,
                sample_rate=request.sample_rate,
                duration=len(cached) / (request.sample_rate * 2),
                processing_time=time.time() - start_time,
                voice_used=voice_id,
            )

        # Prepare voice settings
        voice_settings = self.config.elevenlabs.copy()
        if request.emotion:
            # Adjust settings based on emotion
            voice_settings = self._adjust_for_emotion(voice_settings, request.emotion, request.emotion_strength)

        # Synthesize
        audio_generator = self._client.text_to_speech.convert(
            voice_id=voice_id,
            text=request.text,
            model_id=voice_settings.get("model_id", "eleven_multilingual_v2"),
            voice_settings=voice_settings,
            output_format=request.format,
        )

        audio_data = b"".join(audio_generator)

        # Cache
        self._set_cache(cache_key, audio_data)

        return TTSResult(
            request_id=request.request_id,
            audio_data=audio_data,
            format=request.format,
            sample_rate=request.sample_rate,
            duration=len(audio_data) / (request.sample_rate * 2),
            processing_time=time.time() - start_time,
            voice_used=voice_id,
        )

    async def stream_synthesize(
        self,
        request: TTSRequest,
    ) -> AsyncGenerator[bytes, None]:
        if not self._initialized or not self._client:
            raise RuntimeError("ElevenLabs not initialized")

        voice_id = request.voice_id or request.voice_profile.voice_id if request.voice_profile else "21m00Tcm4TlvDq8ikWAM"
        voice_settings = self.config.elevenlabs.copy()

        if request.emotion:
            voice_settings = self._adjust_for_emotion(voice_settings, request.emotion, request.emotion_strength)

        # ElevenLabs supports streaming
        audio_generator = self._client.text_to_speech.convert(
            voice_id=voice_id,
            text=request.text,
            model_id=voice_settings.get("model_id", "eleven_multilingual_v2"),
            voice_settings=voice_settings,
            output_format=request.format,
            stream=True,
        )

        for chunk in audio_generator:
            yield chunk

    def _adjust_for_emotion(self, settings: dict, emotion: str, strength: float) -> dict:
        """Adjust voice settings for emotion."""
        adjusted = settings.copy()

        emotion_map = {
            "happy": {"stability": 0.3, "similarity_boost": 0.8, "style": 0.5},
            "sad": {"stability": 0.7, "similarity_boost": 0.6, "style": 0.3},
            "angry": {"stability": 0.2, "similarity_boost": 0.9, "style": 0.7},
            "whisper": {"stability": 0.8, "similarity_boost": 0.5, "style": 0.1},
            "excited": {"stability": 0.2, "similarity_boost": 0.8, "style": 0.8},
        }

        if emotion in emotion_map:
            for key, value in emotion_map[emotion].items():
                # Blend with base settings based on strength
                base = adjusted.get(key, value)
                adjusted[key] = base * (1 - strength) + value * strength

        return adjusted

    async def get_available_voices(self) -> list[VoiceProfile]:
        voices = []
        for voice_id, profile in VOICE_REGISTRY.items():
            if profile.provider == TTSProvider.ELEVENLABS:
                voices.append(profile)

        # Also fetch custom voices from API
        if self._initialized and self._client:
            try:
                custom_voices = self._client.voices.get_all()
                for v in custom_voices.voices:
                    voices.append(VoiceProfile(
                        provider=TTSProvider.ELEVENLABS,
                        voice_id=v.voice_id,
                        name=v.name,
                        language="en",
                        description=v.description or "",
                        supports_cloning=True,
                        supports_emotion=True,
                        sample_rate=22050,
                    ))
            except Exception:
                pass

        return voices

    async def shutdown(self) -> None:
        self._initialized = False


# =============================================================================
# COQUI XTTS PROVIDER (Local Voice Cloning)
# =============================================================================

class CoquiXTTSProvider(TTSProviderBase):
    """Coqui XTTS v2 - Local voice cloning, multilingual."""

    def __init__(self, config: TTSConfig):
        super().__init__(config)
        self._model: Any = None
        self._gpt_cond_latent: Any = None
        self._speaker_embedding: Any = None

    async def initialize(self) -> bool:
        try:
            from TTS.api import TTS
            import torch

            model_name = self.config.coqui_xtts.get("model_name", "tts_models/multilingual/multi-dataset/xtts_v2")
            device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"Loading XTTS v2 on {device}...")
            self._model = TTS(model_name).to(device)

            self._initialized = True
            logger.info("CoquiXTTSProvider initialized")
            return True
        except ImportError:
            logger.error("coqui-tts not installed. Install: pip install coqui-tts")
            return False
        except Exception as e:
            logger.exception(f"Coqui XTTS init error: {e}")
            return False

    async def synthesize(self, request: TTSRequest) -> TTSResult:
        if not self._initialized or not self._model:
            raise RuntimeError("Coqui XTTS not initialized")

        start_time = time.time()
        speaker_wav = request.clone_speaker_wav or self.config.coqui_xtts.get("speaker_wav")

        if not speaker_wav:
            raise ValueError("XTTS requires speaker_wav for voice cloning")

        # Check cache
        cache_key = self._get_cache_key(request)
        cached = self._get_cached(cache_key)
        if cached:
            return TTSResult(
                request_id=request.request_id,
                audio_data=cached,
                format=request.format,
                sample_rate=request.sample_rate,
                duration=len(cached) / (request.sample_rate * 2),
                processing_time=time.time() - start_time,
                voice_used="xtts_clone",
            )

        # Synthesize
        loop = asyncio.get_event_loop()
        wav = await loop.run_in_executor(
            None,
            lambda: self._model.tts(
                text=request.text,
                speaker_wav=speaker_wav,
                language=request.language or self.config.coqui_xtts.get("language", "en"),
                temperature=self.config.coqui_xtts.get("temperature", 0.7),
                length_penalty=self.config.coqui_xtts.get("length_penalty", 1.0),
                repetition_penalty=self.config.coqui_xtts.get("repetition_penalty", 1.0),
                top_k=self.config.coqui_xtts.get("top_k", 50),
                top_p=self.config.coqui_xtts.get("top_p", 0.85),
            ),
        )

        # Convert to bytes
        import soundfile as sf
        import io

        buffer = io.BytesIO()
        sf.write(buffer, wav, self.config.coqui_xtts.get("sample_rate", 22050), format=request.format.upper())
        audio_data = buffer.getvalue()

        # Cache
        self._set_cache(cache_key, audio_data)

        return TTSResult(
            request_id=request.request_id,
            audio_data=audio_data,
            format=request.format,
            sample_rate=request.sample_rate,
            duration=len(wav) / 22050,
            processing_time=time.time() - start_time,
            voice_used="xtts_clone",
        )

    async def stream_synthesize(
        self,
        request: TTSRequest,
    ) -> AsyncGenerator[bytes, None]:
        # XTTS doesn't natively stream, synthesize full and chunk
        result = await self.synthesize(request)
        chunk_size = request.chunk_size

        for i in range(0, len(result.audio_data), chunk_size):
            yield result.audio_data[i:i + chunk_size]

    async def get_available_voices(self) -> list[VoiceProfile]:
        return [v for v in VOICE_REGISTRY.values() if v.provider == TTSProvider.COQUI_XTTS]

    async def shutdown(self) -> None:
        self._model = None
        self._initialized = False


# =============================================================================
# PIPER PROVIDER (Fast Local)
# =============================================================================

class PiperProvider(TTSProviderBase):
    """Piper TTS - Fast, lightweight local synthesis."""

    def __init__(self, config: TTSConfig):
        super().__init__(config)
        self._voice: Any = None
        self._config_obj: Any = None

    async def initialize(self) -> bool:
        try:
            from piper import PiperVoice

            model_path = Path(self.config.piper.get("model_path"))
            config_path = Path(self.config.piper.get("config_path"))

            if not model_path.exists():
                logger.error(f"Piper model not found: {model_path}")
                logger.info("Download from: https://github.com/rhasspy/piper/releases")
                return False

            self._voice = PiperVoice.load(str(model_path), str(config_path) if config_path.exists() else None)
            self._initialized = True
            logger.info("PiperProvider initialized")
            return True
        except ImportError:
            logger.error("piper-tts not installed. Install: pip install piper-tts")
            return False
        except Exception as e:
            logger.exception(f"Piper init error: {e}")
            return False

    async def synthesize(self, request: TTSRequest) -> TTSResult:
        if not self._initialized or not self._voice:
            raise RuntimeError("Piper not initialized")

        start_time = time.time()

        # Check cache
        cache_key = self._get_cache_key(request)
        cached = self._get_cached(cache_key)
        if cached:
            return TTSResult(
                request_id=request.request_id,
                audio_data=cached,
                format=request.format,
                sample_rate=request.sample_rate,
                duration=len(cached) / (request.sample_rate * 2),
                processing_time=time.time() - start_time,
                voice_used=request.voice_id or "piper",
            )

        # Synthesize
        import io
        import soundfile as sf

        buffer = io.BytesIO()
        self._voice.synthesize(
            request.text,
            buffer,
            length_scale=self.config.piper.get("length_scale", 1.0),
            noise_scale=self.config.piper.get("noise_scale", 0.667),
            noise_w_scale=self.config.piper.get("noise_w_scale", 0.8),
        )

        # Convert format
        buffer.seek(0)
        data, sr = sf.read(buffer)

        if request.format != "wav":
            out_buffer = io.BytesIO()
            sf.write(out_buffer, data, sr, format=request.format.upper())
            audio_data = out_buffer.getvalue()
        else:
            audio_data = buffer.getvalue()

        # Cache
        self._set_cache(cache_key, audio_data)

        return TTSResult(
            request_id=request.request_id,
            audio_data=audio_data,
            format=request.format,
            sample_rate=sr,
            duration=len(data) / sr,
            processing_time=time.time() - start_time,
            voice_used=request.voice_id or "piper",
        )

    async def stream_synthesize(
        self,
        request: TTSRequest,
    ) -> AsyncGenerator[bytes, None]:
        # Piper can stream via callback
        import io
        import soundfile as sf

        class StreamBuffer:
            def __init__(self):
                self.buffer = io.BytesIO()
                self.chunks = []

            def write(self, data):
                self.chunks.append(data)
                return len(data)

        stream_buf = StreamBuffer()

        def callback(audio_chunk):
            stream_buf.write(audio_chunk.tobytes())

        # This is simplified - real implementation uses Piper's streaming API
        result = await self.synthesize(request)
        chunk_size = request.chunk_size

        for i in range(0, len(result.audio_data), chunk_size):
            yield result.audio_data[i:i + chunk_size]

    async def get_available_voices(self) -> list[VoiceProfile]:
        return [v for v in VOICE_REGISTRY.values() if v.provider == TTSProvider.PIPER]

    async def shutdown(self) -> None:
        self._voice = None
        self._initialized = False


# =============================================================================
# OPENAI TTS PROVIDER
# =============================================================================

class OpenAITTSProvider(TTSProviderBase):
    """OpenAI TTS API provider."""

    def __init__(self, config: TTSConfig):
        super().__init__(config)
        self._client: Any = None
        self._api_key: str | None = None

    async def initialize(self) -> bool:
        try:
            from openai import AsyncOpenAI
            from config import get_settings

            self._api_key = get_settings().openai_api_key
            if not self._api_key:
                logger.warning("OpenAI API key not configured")
                return False

            self._client = AsyncOpenAI(api_key=self._api_key)
            self._initialized = True
            logger.info("OpenAITTSProvider initialized")
            return True
        except ImportError:
            logger.error("openai not installed. Install: pip install openai")
            return False
        except Exception as e:
            logger.exception(f"OpenAI TTS init error: {e}")
            return False

    async def synthesize(self, request: TTSRequest) -> TTSResult:
        if not self._initialized or not self._client:
            raise RuntimeError("OpenAI TTS not initialized")

        start_time = time.time()
        voice = request.voice_id or self.config.openai_tts.get("voice", "alloy")

        # Check cache
        cache_key = self._get_cache_key(request)
        cached = self._get_cached(cache_key)
        if cached:
            return TTSResult(
                request_id=request.request_id,
                audio_data=cached,
                format=request.format,
                sample_rate=request.sample_rate,
                duration=0,
                processing_time=time.time() - start_time,
                voice_used=voice,
            )

        response = await self._client.audio.speech.create(
            model=self.config.openai_tts.get("model", "tts-1-hd"),
            voice=voice,
            input=request.text,
            response_format=request.format,
            speed=request.speed,
        )

        audio_data = response.content

        # Cache
        self._set_cache(cache_key, audio_data)

        return TTSResult(
            request_id=request.request_id,
            audio_data=audio_data,
            format=request.format,
            sample_rate=request.sample_rate,
            duration=0,  # Unknown from API
            processing_time=time.time() - start_time,
            voice_used=voice,
        )

    async def stream_synthesize(
        self,
        request: TTSRequest,
    ) -> AsyncGenerator[bytes, None]:
        if not self._initialized or not self._client:
            raise RuntimeError("OpenAI TTS not initialized")

        voice = request.voice_id or self.config.openai_tts.get("voice", "alloy")

        async with self._client.audio.speech.with_streaming_response.create(
            model=self.config.openai_tts.get("model", "tts-1-hd"),
            voice=voice,
            input=request.text,
            response_format=request.format,
            speed=request.speed,
        ) as response:
            async for chunk in response.iter_bytes(chunk_size=request.chunk_size):
                yield chunk

    async def get_available_voices(self) -> list[VoiceProfile]:
        return [
            VoiceProfile(provider=TTSProvider.OPENAI_TTS, voice_id=v, name=v.title(), language="en")
            for v in ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        ]

    async def shutdown(self) -> None:
        self._initialized = False


# =============================================================================
# TTS ENGINE (Orchestrator)
# =============================================================================

class TTSEngine:
    """
    High-level TTS engine managing provider selection,
    voice profiles, caching, and streaming.
    """

    def __init__(
        self,
        websocket_manager: Any = None,
        event_bus: Any = None,
        config: TTSConfig | None = None,
    ):
        self.websocket_manager = websocket_manager
        self.event_bus = event_bus
        self.config = config or TTSConfig()

        self._providers: dict[TTSProvider, TTSProviderBase] = {}
        self._current_provider: TTSProviderBase | None = None
        self._current_provider_type: TTSProvider = self.config.default_provider
        self._running = False

        # Voice profiles
        self._voices: dict[str, VoiceProfile] = dict(VOICE_REGISTRY)

        # Stats
        self._stats = {
            "total_requests": 0,
            "total_characters": 0,
            "total_audio_seconds": 0.0,
            "avg_latency_ms": 0.0,
            "cache_hits": 0,
            "errors": 0,
        }

        logger.info("TTSEngine initialized")

    async def start(self) -> bool:
        """Initialize all providers and set default."""
        if self._running:
            return True

        # Initialize all available providers
        provider_classes = {
            TTSProvider.EDGE_TTS: EdgeTTSProvider,
            TTSProvider.ELEVENLABS: ElevenLabsProvider,
            TTSProvider.COQUI_XTTS: CoquiXTTSProvider,
            TTSProvider.PIPER: PiperProvider,
            TTSProvider.OPENAI_TTS: OpenAITTSProvider,
        }

        for provider_type, provider_class in provider_classes.items():
            try:
                provider = provider_class(self.config)
                if await provider.initialize():
                    self._providers[provider_type] = provider
                    logger.info(f"TTS Provider ready: {provider_type.value}")
            except Exception as e:
                logger.warning(f"Provider {provider_type.value} failed to init: {e}")

        if not self._providers:
            logger.error("No TTS providers available!")
            return False

        # Set default provider
        if self.config.default_provider in self._providers:
            self._current_provider = self._providers[self.config.default_provider]
            self._current_provider_type = self.config.default_provider
        else:
            # Use first available
            first_type = next(iter(self._providers))
            self._current_provider = self._providers[first_type]
            self._current_provider_type = first_type
            logger.warning(f"Default provider not available, using {first_type.value}")

        self._running = True
        logger.info(f"TTSEngine started with {self._current_provider_type.value}")
        return True

    async def stop(self) -> None:
        """Shutdown all providers."""
        self._running = False

        for provider in self._providers.values():
            await provider.shutdown()

        self._providers.clear()
        self._current_provider = None
        logger.info("TTSEngine stopped")

    async def synthesize(self, request: TTSRequest) -> TTSResult:
        """Synthesize speech using current or specified provider."""
        if not self._running or not self._current_provider:
            raise RuntimeError("TTS engine not running")

        # Determine provider
        provider = self._current_provider
        if request.voice_profile and request.voice_profile.provider in self._providers:
            provider = self._providers[request.voice_profile.provider]

        start_time = time.time()

        try:
            result = await provider.synthesize(request)

            # Update stats
            self._stats["total_requests"] += 1
            self._stats["total_characters"] += len(request.text)
            self._stats["total_audio_seconds"] += result.duration
            latency = (time.time() - start_time) * 1000
            n = self._stats["total_requests"]
            self._stats["avg_latency_ms"] = (self._stats["avg_latency_ms"] * (n - 1) + latency) / n

            # Emit event
            if self.event_bus:
                await self.event_bus.emit("tts.synthesis", {
                    "text_length": len(request.text),
                    "voice": result.voice_used,
                    "duration": result.duration,
                    "latency_ms": latency,
                })

            return result

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"TTS synthesis error: {e}")
            raise

    async def stream_synthesize(
        self,
        request: TTSRequest,
    ) -> AsyncGenerator[bytes, None]:
        """Stream synthesis from current or specified provider."""
        if not self._running or not self._current_provider:
            raise RuntimeError("TTS engine not running")

        provider = self._current_provider
        if request.voice_profile and request.voice_profile.provider in self._providers:
            provider = self._providers[request.voice_profile.provider]

        async for chunk in provider.stream_synthesize(request):
            yield chunk

    async def speak(
        self,
        text: str,
        voice_id: str | None = None,
        emotion: str | None = None,
        language: str | None = None,
        speed: float | None = None,
        stream: bool = True,
    ) -> TTSResult | AsyncGenerator[bytes, None]:
        """Convenience method for simple synthesis."""
        request = TTSRequest(
            text=text,
            voice_id=voice_id,
            emotion=emotion,
            language=language,
            speed=speed or self.config.default_speed,
            stream=stream,
        )

        if stream:
            return self.stream_synthesize(request)
        else:
            return await self.synthesize(request)

    def get_voice(self, voice_id: str) -> VoiceProfile | None:
        """Get voice profile by ID."""
        return self._voices.get(voice_id)

    def list_voices(self, provider: TTSProvider | None = None) -> list[VoiceProfile]:
        """List available voices, optionally filtered by provider."""
        voices = list(self._voices.values())
        if provider:
            voices = [v for v in voices if v.provider == provider]
        return voices

    async def switch_provider(self, provider: TTSProvider) -> bool:
        """Switch active TTS provider."""
        if provider in self._providers:
            self._current_provider = self._providers[provider]
            self._current_provider_type = provider
            logger.info(f"Switched TTS provider to {provider.value}")
            return True
        return False

    def get_stats(self) -> dict[str, Any]:
        return self._stats.copy()

    def get_provider_status(self) -> dict[str, bool]:
        return {p.value: p in self._providers for p in TTSProvider}


# =============================================================================
# FACTORY
# =============================================================================

async def create_tts_engine(
    websocket_manager: Any = None,
    event_bus: Any = None,
    provider: str | TTSProvider | None = None,
) -> TTSEngine:
    """Factory to create and start TTS engine."""
    config = TTSConfig()
    if provider:
        config.default_provider = TTSProvider(provider) if isinstance(provider, str) else provider

    engine = TTSEngine(
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
    "TTSEngine",
    "TTSProvider",
    "TTSConfig",
    "TTSRequest",
    "TTSResult",
    "VoiceProfile",
    "create_tts_engine",
    "EdgeTTSProvider",
    "ElevenLabsProvider",
    "CoquiXTTSProvider",
    "PiperProvider",
    "OpenAITTSProvider",
    "TTSProviderBase",
    "VOICE_REGISTRY",
]