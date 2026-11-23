"""
================================================================================
VEGA SENSES - Perception Layer Package
================================================================================
Multi-modal perception engines for VEGA:
- Voice: Wake word, STT, TTS, Full-duplex voice manager
- Vision: Webcam object detection, face recognition, gestures
- Screen: Capture, OCR, UI parsing
- Gesture: Hand tracking and gesture recognition
================================================================================
"""

from .wake_word import WakeWordEngine, WakeWordDetector
from .stt_engine import STTEngine, StreamingTranscriber
from .tts_engine import TTSEngine, TTSProvider, VoiceProfile
from .voice_manager import VoiceManager, VoiceSession, VoiceState
from .vision_engine import VisionEngine, VisionPipeline, VisionConfig
from .screen_reader import ScreenReader, ScreenCapturer, OCRProcessor
from .gesture_engine import GestureEngine, HandTracker, GestureRecognizer

__all__ = [
    "WakeWordEngine",
    "WakeWordDetector",
    "STTEngine",
    "StreamingTranscriber",
    "TTSEngine",
    "TTSProvider",
    "VoiceProfile",
    "VoiceManager",
    "VoiceSession",
    "VoiceState",
    "VisionEngine",
    "VisionPipeline",
    "VisionConfig",
    "ScreenReader",
    "ScreenCapturer",
    "OCRProcessor",
    "GestureEngine",
    "HandTracker",
    "GestureRecognizer",
]