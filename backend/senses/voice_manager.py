"""
JARVIS OMEGA - Voice Manager
Full-duplex voice pipeline integrating wake word, STT, TTS with barge-in support.
Orchestrates: Wake Word -> STT -> LLM -> TTS with interruption handling.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Optional

import numpy as np

from config import get_settings, settings
from senses.wake_word import WakeWordEngine, DetectionResult
from senses.stt_engine import STTEngine, StreamingTranscriber, TranscriptionResult, TranscriptionSegment
from senses.tts_engine import TTSEngine, TTSRequest, TTSResult, TTSProvider

logger = logging.getLogger("jarvis.senses.voice_manager")


# =============================================================================
# DATA MODELS
# =============================================================================

class VoiceState(str, Enum):
    IDLE = "idle"                    # Waiting for wake word
    LISTENING = "listening"          # Wake word detected, recording
    PROCESSING = "processing"        # STT complete, LLM thinking
    SPEAKING = "speaking"            # TTS playing response
    INTERRUPTED = "interrupted"      # User interrupted (barge-in)


@dataclass
class VoiceSession:
    """Active voice interaction session."""
    session_id: str
    state: VoiceState = VoiceState.IDLE
    wake_word: str | None = None
    wake_word_confidence: float = 0.0

    # Audio buffers
    input_buffer: deque = field(default_factory=lambda: deque(maxlen=100))
    current_utterance: list[np.ndarray] = field(default_factory=list)

    # Transcription
    transcription_segments: list[TranscriptionSegment] = field(default_factory=list)
    final_transcript: str = ""
    language: str | None = None

    # Response
    response_text: str = ""
    response_chunks: list[str] = field(default_factory=list)
    tts_request_id: str | None = None

    # Timing
    started_at: float = field(default_factory=time.time)
    wake_word_at: float | None = None
    stt_started_at: float | None = None
    llm_started_at: float | None = None
    tts_started_at: float | None = None
    first_audio_at: float | None = None

    # Interruption
    interrupted: bool = False
    interruption_time: float | None = None

    def reset(self) -> None:
        """Reset session for new interaction."""
        self.state = VoiceState.IDLE
        self.wake_word = None
        self.wake_word_confidence = 0.0
        self.input_buffer.clear()
        self.current_utterance.clear()
        self.transcription_segments.clear()
        self.final_transcript = ""
        self.language = None
        self.response_text = ""
        self.response_chunks.clear()
        self.tts_request_id = None
        self.interrupted = False
        self.interruption_time = None


@dataclass
class VoiceConfig:
    """Voice manager configuration."""
    # Wake word
    wake_word_enabled: bool = True
    wake_word_cooldown: float = 2.0

    # STT
    stt_streaming: bool = True
    stt_silence_timeout: float = 1.5
    stt_max_utterance: float = 30.0

    # LLM/Orchestrator
    llm_stream_response: bool = True
    llm_timeout: float = 30.0

    # TTS
    tts_streaming: bool = True
    tts_auto_play: bool = True
    tts_emotion_from_context: bool = True

    # Barge-in
    barge_in_enabled: bool = True
    barge_in_sensitivity: float = 0.5  # VAD threshold during playback
    barge_in_min_duration: float = 0.3  # Min speech to trigger barge-in

    # Audio
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1280

    # Playback
    playback_device: int | None = None
    playback_volume: float = 1.0


# =============================================================================
# BARGE-IN HANDLER
# =============================================================================

class BargeInHandler:
    """
    Handles barge-in (interruption) detection during TTS playback.
    Uses VAD on microphone input while audio is playing.
    """

    def __init__(
        self,
        sensitivity: float = 0.5,
        min_speech_duration: float = 0.3,
        sample_rate: int = 16000,
    ):
        self.sensitivity = sensitivity
        self.min_speech_duration = min_speech_duration
        self.sample_rate = sample_rate

        self._vad_model: Any = None
        self._monitoring = False
        self._speech_start: float | None = None
        self._speech_frames: int = 0
        self._callback: Callable[[], Any] | None = None

    async def initialize(self) -> bool:
        """Load VAD model."""
        try:
            import torch
            self._vad_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            logger.info("BargeInHandler VAD loaded")
            return True
        except Exception as e:
            logger.warning(f"BargeIn VAD not available: {e}")
            return False

    def start_monitoring(self, callback: Callable[[], Any]) -> None:
        """Start monitoring for barge-in."""
        self._callback = callback
        self._monitoring = True
        self._speech_start = None
        self._speech_frames = 0

    def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self._monitoring = False
        self._callback = None

    def process_audio(self, audio: np.ndarray) -> bool:
        """
        Process audio chunk for barge-in detection.
        Returns True if barge-in detected.
        """
        if not self._monitoring or self._vad_model is None:
            return False

        try:
            import torch

            if audio.dtype == np.int16:
                audio_tensor = torch.from_numpy(audio.astype(np.float32) / 32768.0)
            else:
                audio_tensor = torch.from_numpy(audio.astype(np.float32))

            # Silero expects 512 samples at 16kHz
            if len(audio_tensor) != 512:
                if len(audio_tensor) > 512:
                    audio_tensor = audio_tensor[:512]
                else:
                    padding = torch.zeros(512 - len(audio_tensor))
                    audio_tensor = torch.cat([audio_tensor, padding])

            with torch.no_grad():
                speech_prob = self._vad_model(audio_tensor, self.sample_rate).item()

            is_speech = speech_prob > self.sensitivity

            if is_speech:
                if self._speech_start is None:
                    self._speech_start = time.time()
                self._speech_frames += 1

                # Check if sustained speech long enough
                speech_duration = self._speech_frames * 512 / self.sample_rate
                if speech_duration >= self.min_speech_duration:
                    if self._callback:
                        self._callback()
                    return True
            else:
                # Reset on silence
                self._speech_start = None
                self._speech_frames = 0

        except Exception:
            pass

        return False


# =============================================================================
# AUDIO PLAYBACK
# =============================================================================

class AudioPlayback:
    """Async audio playback with interruption support."""

    def __init__(
        self,
        device_index: int | None = None,
        sample_rate: int = 22050,
        channels: int = 1,
    ):
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.channels = channels

        self._stream: Any = None
        self._pyaudio: Any = None
        self._playing = False
        self._interrupted = False
        self._current_task: asyncio.Task | None = None

    async def start(self) -> bool:
        """Initialize playback."""
        try:
            import pyaudio

            self._pyaudio = pyaudio.PyAudio()

            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                output_device_index=self.device_index,
                frames_per_buffer=1024,
            )

            logger.info("AudioPlayback initialized")
            return True
        except ImportError:
            logger.error("PyAudio not installed")
            return False
        except Exception as e:
            logger.exception(f"AudioPlayback init error: {e}")
            return False

    async def play(self, audio_data: bytes, format: str = "wav") -> None:
        """Play audio data (blocking until complete or interrupted)."""
        if self._playing:
            logger.warning("Already playing, stopping current")
            await self.stop()

        self._playing = True
        self._interrupted = False

        try:
            # Convert to int16 if needed
            import soundfile as sf
            import io

            buffer = io.BytesIO(audio_data)
            data, sr = sf.read(buffer)

            if sr != self.sample_rate:
                # Resample
                import scipy.signal
                data = scipy.signal.resample(data, int(len(data) * self.sample_rate / sr))

            # Convert to int16
            if data.dtype == np.float32:
                data = (data * 32767).astype(np.int16)

            # Play in chunks
            chunk_size = 1024
            for i in range(0, len(data), chunk_size):
                if self._interrupted:
                    break

                chunk = data[i:i + chunk_size]
                if len(chunk) > 0:
                    self._stream.write(chunk.tobytes())
                    await asyncio.sleep(0)  # Yield control

        except Exception as e:
            logger.error(f"Playback error: {e}")
        finally:
            self._playing = False

    async def play_stream(self, audio_stream: AsyncGenerator[bytes, None]) -> None:
        """Play streaming audio chunks."""
        if self._playing:
            await self.stop()

        self._playing = True
        self._interrupted = False

        try:
            import soundfile as sf
            import io

            async for chunk in audio_stream:
                if self._interrupted:
                    break

                # Decode chunk
                buffer = io.BytesIO(chunk)
                try:
                    data, sr = sf.read(buffer)
                    if sr != self.sample_rate:
                        import scipy.signal
                        data = scipy.signal.resample(data, int(len(data) * self.sample_rate / sr))

                    if data.dtype == np.float32:
                        data = (data * 32767).astype(np.int16)

                    self._stream.write(data.tobytes())
                    await asyncio.sleep(0)
                except Exception:
                    # Might be header chunk, skip
                    pass

        except Exception as e:
            logger.error(f"Stream playback error: {e}")
        finally:
            self._playing = False

    async def stop(self) -> None:
        """Stop playback immediately."""
        self._interrupted = True
        self._playing = False

    async def shutdown(self) -> None:
        await self.stop()
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pyaudio:
            self._pyaudio.terminate()

    @property
    def is_playing(self) -> bool:
        return self._playing


# =============================================================================
# VOICE MANAGER (Main Orchestrator)
# =============================================================================

class VoiceManager:
    """
    Complete voice interaction pipeline manager.
    Orchestrates wake word -> STT -> LLM -> TTS with full-duplex barge-in.
    """

    def __init__(
        self,
        websocket_manager: Any = None,
        event_bus: Any = None,
        orchestrator: Any = None,
        config: VoiceConfig | None = None,
    ):
        self.websocket_manager = websocket_manager
        self.event_bus = event_bus
        self.orchestrator = orchestrator

        self.config = config or VoiceConfig()

        # Sub-engines
        self.wake_word_engine: WakeWordEngine | None = None
        self.stt_engine: STTEngine | None = None
        self.tts_engine: TTSEngine | None = None

        # Components
        self.barge_in_handler = BargeInHandler(
            sensitivity=self.config.barge_in_sensitivity,
            min_speech_duration=self.config.barge_in_min_duration,
            sample_rate=self.config.sample_rate,
        )
        self.audio_playback = AudioPlayback(
            device_index=self.config.playback_device,
            sample_rate=22050,  # TTS output rate
        )

        # Session management
        self.current_session: VoiceSession | None = None
        self._running = False
        self._main_task: asyncio.Task | None = None
        self._audio_capture_task: asyncio.Task | None = None

        # Callbacks
        self._on_wake_word: Callable[[DetectionResult], Any] | None = None
        self._on_transcript: Callable[[TranscriptionResult], Any] | None = None
        self._on_response: Callable[[str], Any] | None = None
        self._on_speech_start: Callable[[], Any] | None = None
        self._on_speech_end: Callable[[], Any] | None = None

        logger.info("VoiceManager initialized")

    async def start(
        self,
        wake_word_engine: WakeWordEngine | None = None,
        stt_engine: STTEngine | None = None,
        tts_engine: TTSEngine | None = None,
    ) -> bool:
        """Start the voice manager with engines."""
        if self._running:
            return True

        self.wake_word_engine = wake_word_engine
        self.stt_engine = stt_engine
        self.tts_engine = tts_engine

        # Initialize engines if not provided
        if not self.wake_word_engine:
            self.wake_word_engine = WakeWordEngine(
                websocket_manager=self.websocket_manager,
                event_bus=self.event_bus,
            )
            await self.wake_word_engine.start()

        if not self.stt_engine:
            self.stt_engine = STTEngine(
                websocket_manager=self.websocket_manager,
                event_bus=self.event_bus,
            )
            await self.stt_engine.start()

        if not self.tts_engine:
            self.tts_engine = TTSEngine(
                websocket_manager=self.websocket_manager,
                event_bus=self.event_bus,
            )
            await self.tts_engine.start()

        # Initialize components
        await self.barge_in_handler.initialize()
        await self.audio_playback.start()

        # Register wake word callback
        self.wake_word_engine.on_detection(self._on_wake_word_detected)

        # Start main loop
        self._running = True
        self._main_task = asyncio.create_task(self._main_loop())
        self._audio_capture_task = asyncio.create_task(self._audio_capture_loop())

        logger.info("VoiceManager started")
        return True

    async def stop(self) -> None:
        """Stop the voice manager."""
        self._running = False

        if self._main_task:
            self._main_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._main_task

        if self._audio_capture_task:
            self._audio_capture_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._audio_capture_task

        # Stop engines
        if self.wake_word_engine:
            await self.wake_word_engine.stop()
        if self.stt_engine:
            await self.stt_engine.stop()
        if self.tts_engine:
            await self.tts_engine.stop()

        await self.audio_playback.shutdown()

        logger.info("VoiceManager stopped")

    # -------------------------------------------------------------------------
    # CALLBACK REGISTRATION
    # -------------------------------------------------------------------------

    def on_wake_word(self, callback: Callable[[DetectionResult], Any]) -> None:
        self._on_wake_word = callback

    def on_transcript(self, callback: Callable[[TranscriptionResult], Any]) -> None:
        self._on_transcript = callback

    def on_response(self, callback: Callable[[str], Any]) -> None:
        self._on_response = callback

    def on_speech_start(self, callback: Callable[[], Any]) -> None:
        self._on_speech_start = callback

    def on_speech_end(self, callback: Callable[[], Any]) -> None:
        self._on_speech_end = callback

    # -------------------------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------------------------

    async def _main_loop(self) -> None:
        """Main state machine loop."""
        while self._running:
            try:
                if self.current_session:
                    await self._process_session()

                await asyncio.sleep(0.05)  # 20Hz loop

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Voice main loop error: {e}")
                await asyncio.sleep(0.1)

    async def _process_session(self) -> None:
        """Process current session based on state."""
        session = self.current_session
        if not session:
            return

        if session.state == VoiceState.IDLE:
            # Waiting for wake word - handled by callback
            pass

        elif session.state == VoiceState.LISTENING:
            # STT is running, check for completion
            if session.final_transcript and not session.stt_started_at:
                # STT completed, send to orchestrator
                await self._send_to_orchestrator(session)

        elif session.state == VoiceState.PROCESSING:
            # Waiting for LLM response
            if session.response_text and not session.tts_started_at:
                # LLM response received, start TTS
                await self._start_tts(session)

        elif session.state == VoiceState.SPEAKING:
            # TTS playing, check for barge-in
            if session.interrupted:
                await self._handle_interruption(session)

        elif session.state == VoiceState.INTERRUPTED:
            # Handle interruption, go back to listening
            await self._recover_from_interruption(session)

    # -------------------------------------------------------------------------
    # AUDIO CAPTURE LOOP
    # -------------------------------------------------------------------------

    async def _audio_capture_loop(self) -> None:
        """Continuous audio capture feeding wake word + STT + barge-in."""
        try:
            import pyaudio

            pyaudio_instance = pyaudio.PyAudio()
            stream = pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=None,
                frames_per_buffer=self.config.chunk_size,
            )

            stream.start_stream()

            while self._running:
                try:
                    data = stream.read(self.config.chunk_size, exception_on_overflow=False)
                    audio_chunk = np.frombuffer(data, dtype=np.int16)

                    # Feed to wake word engine
                    if self.wake_word_engine and self.current_session and self.current_session.state == VoiceState.IDLE:
                        self.wake_word_engine.feed_audio(audio_chunk)

                    # Feed to STT if listening
                    if self.current_session and self.current_session.state == VoiceState.LISTENING:
                        # This is handled by StreamingTranscriber internally
                        pass

                    # Check barge-in during playback
                    if (self.config.barge_in_enabled and
                        self.current_session and
                        self.current_session.state == VoiceState.SPEAKING and
                        self.audio_playback.is_playing):

                        if self.barge_in_handler.process_audio(audio_chunk):
                            self.current_session.interrupted = True
                            self.current_session.interruption_time = time.time()
                            logger.info("Barge-in detected!")

                    await asyncio.sleep(0)

                except Exception as e:
                    logger.error(f"Audio capture error: {e}")
                    await asyncio.sleep(0.1)

            stream.stop_stream()
            stream.close()
            pyaudio_instance.terminate()

        except Exception as e:
            logger.exception(f"Audio capture loop failed: {e}")

    # -------------------------------------------------------------------------
    # WAKE WORD HANDLING
    # -------------------------------------------------------------------------

    async def _on_wake_word_detected(self, result: DetectionResult) -> None:
        """Handle wake word detection."""
        if not self.current_session:
            self.current_session = VoiceSession(session_id=str(uuid.uuid4()))

        session = self.current_session

        # Check cooldown
        if session.wake_word_at and time.time() - session.wake_word_at < self.config.wake_word_cooldown:
            return

        session.state = VoiceState.LISTENING
        session.wake_word = result.wake_word
        session.wake_word_confidence = result.confidence
        session.wake_word_at = time.time()

        logger.info(f"Wake word: {result.wake_word} (conf: {result.confidence:.3f})")

        # Notify callbacks
        if self._on_wake_word:
            await self._on_wake_word(result)

        # Broadcast to HUD
        if self.websocket_manager:
            await self.websocket_manager.broadcast({
                "type": "voice_state",
                "payload": {"state": "listening", "wake_word": result.wake_word},
            }, room="hud")

        # Start STT streaming
        if self.stt_engine:
            await self._start_stt_streaming(session)

    async def _start_stt_streaming(self, session: VoiceSession) -> None:
        """Start streaming STT for the session."""
        session.stt_started_at = time.time()

        def on_segment(segment: TranscriptionSegment):
            session.transcription_segments.append(segment)

            if segment.is_final:
                session.final_transcript += " " + segment.text
                session.final_transcript = session.final_transcript.strip()
                session.language = segment.language

                # Broadcast interim
                if self.websocket_manager:
                    self.websocket_manager.broadcast({
                        "type": "stt_final",
                        "payload": {
                            "text": segment.text,
                            "full_transcript": session.final_transcript,
                            "language": segment.language,
                        },
                    }, room="hud")

        # Start STT stream session
        self._stt_stream_session_id = await self.stt_engine.start_stream_session(
            session_id=f"stt_{session.session_id}",
            on_segment=on_segment,
        )

    # -------------------------------------------------------------------------
    # ORCHESTRATOR INTEGRATION
    # -------------------------------------------------------------------------

    async def _send_to_orchestrator(self, session: VoiceSession) -> None:
        """Send transcript to orchestrator for processing."""
        session.state = VoiceState.PROCESSING
        session.llm_started_at = time.time()

        if self.websocket_manager:
            await self.websocket_manager.broadcast({
                "type": "voice_state",
                "payload": {"state": "processing"},
            }, room="hud")

        if not self.orchestrator or not session.final_transcript:
            return

        try:
            # Send to orchestrator with streaming response
            async for chunk in self.orchestrator.chat_stream(
                session.final_transcript,
                context={"voice_session_id": session.session_id, "language": session.language},
            ):
                if session.interrupted:
                    break

                session.response_chunks.append(chunk)
                session.response_text += chunk

                # If we have enough for a sentence, start TTS
                if self._should_start_tts(session.response_text):
                    await self._start_tts(session)
                    break

            # If no TTS started yet, start with full response
            if session.response_text and session.state == VoiceState.PROCESSING:
                await self._start_tts(session)

        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            session.state = VoiceState.IDLE

    def _should_start_tts(self, text: str) -> bool:
        """Check if we have enough text to start TTS (sentence boundary)."""
        # Start TTS on sentence boundaries
        sentences = text.strip().split('. ')
        return len(sentences) >= 1 and len(text) > 20

    # -------------------------------------------------------------------------
    # TTS HANDLING
    # -------------------------------------------------------------------------

    async def _start_tts(self, session: VoiceSession) -> None:
        """Start TTS playback for response."""
        if not session.response_text or not self.tts_engine:
            return

        session.state = VoiceState.SPEAKING
        session.tts_started_at = time.time()

        # Determine emotion from context
        emotion = None
        if self.config.tts_emotion_from_context:
            emotion = self._detect_emotion(session.response_text)

        # Create TTS request
        tts_request = TTSRequest(
            text=session.response_text,
            emotion=emotion,
            language=session.language,
            stream=self.config.tts_streaming,
        )

        # Broadcast speech start
        if self.websocket_manager:
            await self.websocket_manager.broadcast({
                "type": "voice_state",
                "payload": {"state": "speaking", "text": session.response_text[:100]},
            }, room="hud")

        if self._on_speech_start:
            await self._on_speech_start()

        # Start barge-in monitoring
        self.barge_in_handler.start_monitoring(lambda: self._trigger_barge_in(session))

        try:
            if self.config.tts_streaming:
                # Stream TTS and play simultaneously
                async for chunk in self.tts_engine.stream_synthesize(tts_request):
                    if session.interrupted:
                        break
                    await self.audio_playback.play_stream(
                        self._chunk_generator(chunk)
                    )
            else:
                # Full synthesis then play
                result = await self.tts_engine.synthesize(tts_request)
                await self.audio_playback.play(result.audio_data)

        except Exception as e:
            logger.error(f"TTS playback error: {e}")
        finally:
            # Stop barge-in monitoring
            self.barge_in_handler.stop_monitoring()

            if self._on_speech_end:
                await self._on_speech_end()

            # Session complete
            if not session.interrupted:
                session.state = VoiceState.IDLE
                session.reset()

                if self.websocket_manager:
                    await self.websocket_manager.broadcast({
                        "type": "voice_state",
                        "payload": {"state": "idle"},
                    }, room="hud")

    def _detect_emotion(self, text: str) -> str | None:
        """Simple emotion detection from response text."""
        text_lower = text.lower()

        if any(w in text_lower for w in ["sorry", "apologize", "unfortunate", "sad"]):
            return "sad"
        elif any(w in text_lower for w in ["great", "awesome", "excellent", "happy", "glad"]):
            return "happy"
        elif any(w in text_lower for w in ["warning", "danger", "error", "critical", "stop"]):
            return "angry"
        elif any(w in text_lower for w in ["whisper", "quiet", "secret", "shh"]):
            return "whisper"
        elif any(w in text_lower for w in ["wow", "amazing", "incredible", "fantastic"]):
            return "excited"

        return None

    async def _chunk_generator(self, chunk: bytes) -> AsyncGenerator[bytes, None]:
        """Wrap single chunk in async generator."""
        yield chunk

    # -------------------------------------------------------------------------
    # BARGE-IN HANDLING
    # -------------------------------------------------------------------------

    def _trigger_barge_in(self, session: VoiceSession) -> None:
        """Trigger barge-in interruption."""
        logger.info("Barge-in triggered!")
        session.interrupted = True
        session.interruption_time = time.time()

        # Stop current playback
        asyncio.create_task(self.audio_playback.stop())

        # Broadcast interruption
        if self.websocket_manager:
            self.websocket_manager.broadcast({
                "type": "voice_state",
                "payload": {"state": "interrupted", "reason": "barge_in"},
            }, room="hud")

    async def _handle_interruption(self, session: VoiceSession) -> None:
        """Handle interruption - stop everything, prepare for new input."""
        session.state = VoiceState.INTERRUPTED

        # Stop TTS stream if active
        if hasattr(self, '_stt_stream_session_id'):
            await self.stt_engine.stop_stream_session(self._stt_stream_session_id)

        # Clear response
        session.response_text = ""
        session.response_chunks.clear()

        # Go back to listening after brief pause
        await asyncio.sleep(0.5)
        session.state = VoiceState.LISTENING
        session.final_transcript = ""
        session.transcription_segments.clear()

        # Restart STT
        await self._start_stt_streaming(session)

    async def _recover_from_interruption(self, session: VoiceSession) -> None:
        """Recover from interruption state."""
        if not session.interrupted:
            session.state = VoiceState.LISTENING

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    async def inject_text(self, text: str) -> None:
        """Inject text as if user spoke it (for testing/text input)."""
        if not self.current_session:
            self.current_session = VoiceSession(session_id=str(uuid.uuid4()))

        session = self.current_session
        session.final_transcript = text
        session.state = VoiceState.LISTENING
        await self._send_to_orchestrator(session)

    async def say(self, text: str, emotion: str | None = None) -> None:
        """Make VEGA speak text directly."""
        if not self.tts_engine:
            return

        request = TTSRequest(text=text, emotion=emotion, stream=True)

        if self._on_speech_start:
            await self._on_speech_start()

        self.barge_in_handler.start_monitoring(lambda: None)

        try:
            async for chunk in self.tts_engine.stream_synthesize(request):
                await self.audio_playback.play_stream(self._chunk_generator(chunk))
        finally:
            self.barge_in_handler.stop_monitoring()

            if self._on_speech_end:
                await self._on_speech_end()

    def get_state(self) -> VoiceState:
        return self.current_session.state if self.current_session else VoiceState.IDLE

    def get_session(self) -> VoiceSession | None:
        return self.current_session


# =============================================================================
# FACTORY
# =============================================================================

async def create_voice_manager(
    websocket_manager: Any = None,
    event_bus: Any = None,
    orchestrator: Any = None,
    wake_word_engine: WakeWordEngine | None = None,
    stt_engine: STTEngine | None = None,
    tts_engine: TTSEngine | None = None,
) -> VoiceManager:
    """Factory to create and start complete voice pipeline."""
    manager = VoiceManager(
        websocket_manager=websocket_manager,
        event_bus=event_bus,
        orchestrator=orchestrator,
    )

    await manager.start(
        wake_word_engine=wake_word_engine,
        stt_engine=stt_engine,
        tts_engine=tts_engine,
    )

    return manager


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "VoiceManager",
    "VoiceSession",
    "VoiceState",
    "VoiceConfig",
    "BargeInHandler",
    "AudioPlayback",
    "create_voice_manager",
]