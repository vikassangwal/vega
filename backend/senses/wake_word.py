"""
JARVIS OMEGA - Wake Word Engine
Custom wake word detection using openWakeWord with ONNX runtime.
Supports multiple wake words, VAD integration, and cooldown handling.
"""
from __future__ import annotations

import asyncio
import logging
import struct
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from config import get_settings, settings

logger = logging.getLogger("jarvis.senses.wake_word")


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class WakeWordConfig:
    """Configuration for a single wake word model."""
    name: str
    model_path: str
    threshold: float = 0.5
    cooldown: float = 2.0  # seconds
    enabled: bool = True


@dataclass
class DetectionResult:
    """Result of wake word detection."""
    detected: bool
    wake_word: str | None = None
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)
    audio_chunk: np.ndarray | None = None


@dataclass
class WakeWordStats:
    """Statistics for monitoring."""
    total_detections: int = 0
    false_positives: int = 0
    avg_confidence: float = 0.0
    last_detection: float | None = None
    uptime_seconds: float = 0.0


# =============================================================================
# WAKE WORD DETECTOR (Low-level ONNX inference)
# =============================================================================

class WakeWordDetector:
    """
    Low-level wake word detector using ONNX Runtime.
    Compatible with openWakeWord model format.
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        chunk_size: int = 1280,
    ):
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        self._session: Any = None
        self._input_name: str = ""
        self._output_name: str = ""
        self._initialized = False

        # Audio buffer for context
        self._buffer = deque(maxlen=int(sample_rate * 1.5 / chunk_size))  # 1.5 sec context

    async def initialize(self) -> bool:
        """Load ONNX model and prepare for inference."""
        try:
            import onnxruntime as ort

            if not self.model_path.exists():
                logger.warning(f"Wake word model not found: {self.model_path}")
                return False

            # Create ONNX session with optimized settings
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            self._session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],  # Can add CUDAExecutionProvider
            )

            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name

            self._initialized = True
            logger.info(f"Wake word detector loaded: {self.model_path.name}")
            return True

        except ImportError:
            logger.error("onnxruntime not installed. Install with: pip install onnxruntime")
            return False
        except Exception as e:
            logger.exception(f"Failed to load wake word model: {e}")
            return False

    def process_chunk(self, audio_chunk: np.ndarray) -> tuple[bool, float]:
        """
        Process a single audio chunk and return detection result.

        Args:
            audio_chunk: Int16 or float32 audio data (chunk_size samples)

        Returns:
            (detected, confidence)
        """
        if not self._initialized or self._session is None:
            return False, 0.0

        try:
            # Ensure float32 [-1, 1]
            if audio_chunk.dtype == np.int16:
                audio_chunk = audio_chunk.astype(np.float32) / 32768.0
            elif audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)

            # Add to buffer
            self._buffer.append(audio_chunk)

            # Need enough context for model (typically 1-1.5 seconds)
            if len(self._buffer) < self._buffer.maxlen:
                return False, 0.0

            # Concatenate buffer
            context = np.concatenate(list(self._buffer))

            # Reshape for model: (1, sequence_length, 1) or (1, sequence_length)
            # Model expects specific input shape - adjust based on model
            if context.ndim == 1:
                context = context.reshape(1, -1, 1)

            # Run inference
            outputs = self._session.run([self._output_name], {self._input_name: context})
            confidence = float(outputs[0].flatten()[-1])  # Last timestep

            detected = confidence >= self.threshold
            return detected, confidence

        except Exception as e:
            logger.error(f"Wake word inference error: {e}")
            return False, 0.0

    def reset(self) -> None:
        """Clear audio buffer."""
        self._buffer.clear()


# =============================================================================
# WAKE WORD ENGINE (High-level orchestration)
# =============================================================================

class WakeWordEngine:
    """
    High-level wake word engine managing multiple detectors,
    VAD integration, cooldown handling, and event emission.
    """

    def __init__(
        self,
        websocket_manager: Any = None,
        event_bus: Any = None,
        config: dict | None = None,
    ):
        self.websocket_manager = websocket_manager
        self.event_bus = event_bus

        # Load config from settings
        cfg = config or settings.wake_word.model_dump() if hasattr(settings, 'wake_word') else {}
        self.enabled = cfg.get("enabled", True)
        self.sample_rate = cfg.get("sample_rate", 16000)
        self.chunk_size = cfg.get("chunk_size", 1280)
        self.vad_threshold = cfg.get("vad_threshold", 0.5)
        self.global_cooldown = cfg.get("cooldown", 2.0)

        # Detectors
        self.detectors: dict[str, WakeWordDetector] = {}
        self.detector_configs: dict[str, WakeWordConfig] = {}

        # VAD
        self._vad_model: Any = None
        self._vad_enabled = True

        # State
        self._running = False
        self._audio_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._process_task: asyncio.Task | None = None
        self._last_detection_time: dict[str, float] = {}
        self._stats = WakeWordStats()
        self._start_time = time.time()

        # Callbacks
        self._on_detection: list[Callable[[DetectionResult], Any]] = []

        logger.info("WakeWordEngine initialized")

    async def start(self) -> bool:
        """Start the wake word engine."""
        if self._running:
            return True

        if not self.enabled:
            logger.info("Wake word engine disabled in config")
            return False

        # Initialize VAD
        await self._init_vad()

        # Load wake word models
        await self._load_models()

        if not self.detectors:
            logger.warning("No wake word models loaded, engine will not detect")
            return False

        # Start processing loop
        self._running = True
        self._process_task = asyncio.create_task(self._process_loop())

        logger.info("WakeWordEngine started")
        return True

    async def stop(self) -> None:
        """Stop the wake word engine."""
        self._running = False

        if self._process_task:
            self._process_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._process_task

        # Clear queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        logger.info("WakeWordEngine stopped")

    async def _init_vad(self) -> None:
        """Initialize Silero VAD."""
        try:
            import torch
            import silero_vad

            self._vad_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            logger.info("Silero VAD loaded")
        except Exception as e:
            logger.warning(f"Could not load Silero VAD: {e}. VAD disabled.")
            self._vad_enabled = False

    async def _load_models(self) -> None:
        """Load all configured wake word models."""
        # Built-in models (shipped with openWakeWord)
        model_dir = Path(__file__).parent.parent.parent / "models" / "wakeword"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Default models to try
        default_models = [
            ("jarvis", "jarvis.onnx"),
            ("omega", "omega.onnx"),
            ("hey_jarvis", "hey_jarvis.onnx"),
            ("hey_omega", "hey_omega.onnx"),
        ]

        # Custom models from config
        custom_models = {}
        if hasattr(settings, 'wake_word'):
            custom_models = getattr(settings.wake_word, 'custom_models', {})

        all_models = list(default_models) + [(name, path) for name, path in custom_models.items()]

        for name, model_file in all_models:
            model_path = model_dir / model_file

            # If model doesn't exist, try to download or skip
            if not model_path.exists():
                logger.debug(f"Wake word model not found: {model_path}")
                continue

            config = WakeWordConfig(
                name=name,
                model_path=str(model_path),
                threshold=0.5,  # Can be per-model
                cooldown=self.global_cooldown,
            )

            detector = WakeWordDetector(
                model_path=str(model_path),
                threshold=config.threshold,
                sample_rate=self.sample_rate,
                chunk_size=self.chunk_size,
            )

            if await detector.initialize():
                self.detectors[name] = detector
                self.detector_configs[name] = config
                self._last_detection_time[name] = 0.0
                logger.info(f"Loaded wake word: {name}")

    def feed_audio(self, audio_data: bytes | np.ndarray) -> None:
        """Feed audio data to processing queue (non-blocking)."""
        if not self._running:
            return

        try:
            # Convert to numpy if bytes
            if isinstance(audio_data, bytes):
                audio_data = np.frombuffer(audio_data, dtype=np.int16)

            self._audio_queue.put_nowait(audio_data)
        except asyncio.QueueFull:
            logger.warning("Wake word audio queue full, dropping chunk")

    async def _process_loop(self) -> None:
        """Main processing loop."""
        chunk_samples = self.chunk_size
        buffer = np.array([], dtype=np.int16)

        while self._running:
            try:
                # Get audio chunk with timeout
                try:
                    chunk = await asyncio.wait_for(
                        self._audio_queue.get(),
                        timeout=0.1,
                    )
                except asyncio.TimeoutError:
                    continue

                # Accumulate to chunk size
                buffer = np.concatenate([buffer, chunk])

                while len(buffer) >= chunk_samples:
                    process_chunk = buffer[:chunk_samples]
                    buffer = buffer[chunk_samples:]

                    # Run VAD first (quick check)
                    if self._vad_enabled and self._vad_model is not None:
                        is_speech = await self._run_vad(process_chunk)
                        if not is_speech:
                            continue

                    # Run wake word detection on all models
                    for name, detector in self.detectors.items():
                        if not self.detector_configs[name].enabled:
                            continue

                        # Check cooldown
                        if time.time() - self._last_detection_time[name] < self.detector_configs[name].cooldown:
                            continue

                        detected, confidence = detector.process_chunk(process_chunk)

                        if detected:
                            await self._handle_detection(name, confidence, process_chunk)
                            self._last_detection_time[name] = time.time()
                            break  # Only trigger one wake word per chunk

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Wake word processing error: {e}")
                await asyncio.sleep(0.1)

    async def _run_vad(self, audio_chunk: np.ndarray) -> bool:
        """Run Silero VAD on audio chunk."""
        try:
            import torch

            # Convert to float32 tensor
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
                speech_prob = self._vad_model(audio_tensor, self.sample_rate).item()

            return speech_prob > self.vad_threshold

        except Exception:
            return True  # Default to processing if VAD fails

    async def _handle_detection(
        self,
        wake_word: str,
        confidence: float,
        audio_chunk: np.ndarray,
    ) -> None:
        """Handle wake word detection event."""
        result = DetectionResult(
            detected=True,
            wake_word=wake_word,
            confidence=confidence,
            timestamp=time.time(),
            audio_chunk=audio_chunk,
        )

        # Update stats
        self._stats.total_detections += 1
        self._stats.last_detection = time.time()
        self._stats.avg_confidence = (
            (self._stats.avg_confidence * (self._stats.total_detections - 1) + confidence)
            / self._stats.total_detections
        )

        logger.info(f"Wake word detected: {wake_word} (confidence: {confidence:.3f})")

        # Emit event
        if self.event_bus:
            await self.event_bus.emit("wake_word.detected", {
                "wake_word": wake_word,
                "confidence": confidence,
                "timestamp": result.timestamp,
            })

        # Notify WebSocket clients
        if self.websocket_manager:
            await self.websocket_manager.broadcast({
                "type": "wake_word",
                "payload": {
                    "wake_word": wake_word,
                    "confidence": confidence,
                    "timestamp": result.timestamp,
                },
            }, room="hud")

        # Call registered callbacks
        for callback in self._on_detection:
            try:
                await callback(result)
            except Exception as e:
                logger.error(f"Wake word callback error: {e}")

    def on_detection(self, callback: Callable[[DetectionResult], Any]) -> None:
        """Register a callback for wake word detections."""
        self._on_detection.append(callback)

    def set_threshold(self, wake_word: str, threshold: float) -> bool:
        """Update detection threshold for a wake word."""
        if wake_word in self.detectors:
            self.detectors[wake_word].threshold = threshold
            self.detector_configs[wake_word].threshold = threshold
            return True
        return False

    def enable_wake_word(self, wake_word: str, enabled: bool = True) -> bool:
        """Enable/disable a specific wake word."""
        if wake_word in self.detector_configs:
            self.detector_configs[wake_word].enabled = enabled
            return True
        return False

    def get_stats(self) -> WakeWordStats:
        """Get engine statistics."""
        self._stats.uptime_seconds = time.time() - self._start_time
        return self._stats

    def get_available_wake_words(self) -> list[str]:
        """Get list of loaded wake words."""
        return list(self.detectors.keys())


# =============================================================================
# AUDIO CAPTURE HELPER
# =============================================================================

class AudioCapture:
    """
    High-level audio capture using PyAudio or sounddevice.
    Feeds directly into WakeWordEngine.
    """

    def __init__(
        self,
        wake_word_engine: WakeWordEngine,
        device_index: int | None = None,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1280,
    ):
        self.engine = wake_word_engine
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size

        self._stream: Any = None
        self._pyaudio: Any = None
        self._running = False
        self._capture_task: asyncio.Task | None = None

    async def start(self) -> bool:
        """Start audio capture."""
        try:
            import pyaudio

            self._pyaudio = pyaudio.PyAudio()

            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback,
            )

            self._running = True
            self._stream.start_stream()
            logger.info("Audio capture started")
            return True

        except ImportError:
            logger.error("PyAudio not installed. Install with: pip install pyaudio")
            return False
        except Exception as e:
            logger.exception(f"Failed to start audio capture: {e}")
            return False

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback (runs in separate thread)."""
        if self._running and in_data:
            # Feed to engine (thread-safe via queue)
            try:
                self.engine.feed_audio(in_data)
            except Exception:
                pass
        return (None, pyaudio.paContinue)

    async def stop(self) -> None:
        """Stop audio capture."""
        self._running = False

        if self._stream:
            self._stream.stop_stream()
            self._stream.close()

        if self._pyaudio:
            self._pyaudio.terminate()

        logger.info("Audio capture stopped")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

async def create_wake_word_engine(
    websocket_manager: Any = None,
    event_bus: Any = None,
) -> WakeWordEngine:
    """Factory to create and start wake word engine."""
    engine = WakeWordEngine(
        websocket_manager=websocket_manager,
        event_bus=event_bus,
    )
    await engine.start()
    return engine


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "WakeWordEngine",
    "WakeWordDetector",
    "WakeWordConfig",
    "DetectionResult",
    "WakeWordStats",
    "AudioCapture",
    "create_wake_word_engine",
]