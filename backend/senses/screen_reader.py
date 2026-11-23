"""
JARVIS OMEGA - Screen Reader
Advanced screen perception: multi-monitor capture, OCR, UI element detection,
window tracking, change detection, and accessibility integration.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Optional

import cv2
import numpy as np

from config import get_settings, settings

logger = logging.getLogger("jarvis.senses.screen_reader")


# =============================================================================
# DATA MODELS
# =============================================================================

class ScreenRegionType(str, Enum):
    FULL = "full"
    WINDOW = "window"
    REGION = "region"
    MONITOR = "monitor"


@dataclass
class ScreenRegion:
    """Screen region definition."""
    x: int
    y: int
    width: int
    height: int
    monitor: int = 0
    name: str = ""

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "monitor": self.monitor,
            "name": self.name,
        }

    def to_mss(self) -> dict:
        return {
            "left": self.x,
            "top": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class UIElement:
    """UI element detected on screen."""
    element_id: str
    element_type: str  # button, input, text, image, link, etc.
    bbox: ScreenRegion
    text: str = ""
    confidence: float = 1.0
    attributes: dict = field(default_factory=dict)
    # Accessibility
    role: str = ""
    name: str = ""
    state: str = ""
    # Interaction
    clickable: bool = False
    focused: bool = False
    enabled: bool = True


@dataclass
class ScreenCaptureResult:
    """Result of screen capture + analysis."""
    capture_id: str
    timestamp: float
    monitor: int
    region: ScreenRegion | None
    frame: np.ndarray
    width: int
    height: int
    # Analysis
    text_blocks: list[UIElement] = field(default_factory=list)
    ui_elements: list[UIElement] = field(default_factory=list)
    changes: list[ScreenRegion] = field(default_factory=list)
    processing_time: float = 0.0


@dataclass
class WindowInfo:
    """Window/application information."""
    window_id: str
    title: str
    app_name: str
    pid: int
    bounds: ScreenRegion
    is_active: bool
    is_visible: bool
    z_order: int


@dataclass
class ScreenReaderConfig:
    """Screen reader configuration."""
    # Capture
    capture_fps: int = 10
    monitors: list[int] = field(default_factory=lambda: [0])
    capture_cursor: bool = True

    # OCR
    ocr_enabled: bool = True
    ocr_engine: str = "tesseract"  # tesseract, easyocr, paddleocr
    ocr_languages: list[str] = field(default_factory=lambda: ["eng"])
    ocr_confidence_threshold: float = 0.5
    ocr_interval: int = 5  # Run OCR every N frames

    # OmniParser (UI detection)
    omniparser_enabled: bool = False
    omniparser_endpoint: str = "http://vega-screen-parser:8083"
    omniparser_confidence: float = 0.3
    omniparser_interval: int = 10

    # Change detection
    change_detection_enabled: bool = True
    change_threshold: float = 0.05  # Fraction of pixels changed
    change_interval: int = 2

    # Window tracking
    track_windows: bool = True
    window_update_interval: float = 1.0

    # Regions of interest
    monitored_regions: list[ScreenRegion] = field(default_factory=list)

    # Cache
    cache_frames: int = 5


# =============================================================================
# SCREEN CAPTURER (mss-based)
# =============================================================================

class ScreenCapturer:
    """High-performance screen capture using mss."""

    def __init__(self, config: ScreenReaderConfig):
        self.config = config
        self._sct: Any = None
        self._monitors: list[dict] = []
        self._last_frames: dict[int, np.ndarray] = {}
        self._frame_hashes: dict[int, str] = {}

    async def initialize(self) -> bool:
        try:
            import mss
            self._sct = mss.mss()
            self._monitors = self._sct.monitors
            logger.info(f"ScreenCapturer initialized: {len(self._monitors)-1} monitors")
            return True
        except ImportError:
            logger.error("mss not installed. Install: pip install mss")
            return False
        except Exception as e:
            logger.exception(f"ScreenCapturer init error: {e}")
            return False

    def capture_monitor(self, monitor_index: int) -> np.ndarray | None:
        """Capture full monitor."""
        if not self._sct or monitor_index >= len(self._monitors):
            return None

        try:
            monitor = self._monitors[monitor_index]
            screenshot = self._sct.grab(monitor)
            frame = np.array(screenshot)

            if not self.config.capture_cursor:
                # Remove cursor if present (mss includes cursor by default on Windows)
                pass

            # Convert BGRA to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
        except Exception as e:
            logger.error(f"Monitor capture error: {e}")
            return None

    def capture_region(self, region: ScreenRegion) -> np.ndarray | None:
        """Capture specific region."""
        if not self._sct:
            return None

        try:
            monitor = region.to_mss()
            screenshot = self._sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
        except Exception as e:
            logger.error(f"Region capture error: {e}")
            return None

    def capture_all_monitors(self) -> dict[int, np.ndarray]:
        """Capture all configured monitors."""
        frames = {}
        for monitor_idx in self.config.monitors:
            frame = self.capture_monitor(monitor_idx)
            if frame is not None:
                frames[monitor_idx] = frame
        return frames

    def detect_changes(self, monitor_index: int, current_frame: np.ndarray) -> list[ScreenRegion]:
        """Detect changed regions since last frame."""
        changes = []

        if monitor_index not in self._last_frames:
            self._last_frames[monitor_index] = current_frame.copy()
            return changes

        try:
            # Compute difference
            prev = self._last_frames[monitor_index]
            if prev.shape != current_frame.shape:
                self._last_frames[monitor_index] = current_frame.copy()
                return changes

            # Grayscale for comparison
            prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

            # Absolute difference
            diff = cv2.absdiff(prev_gray, curr_gray)

            # Threshold
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

            # Find contours of changes
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            total_pixels = current_frame.shape[0] * current_frame.shape[1]
            min_change_pixels = total_pixels * self.config.change_threshold

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_change_pixels:
                    x, y, w, h = cv2.boundingRect(contour)
                    changes.append(ScreenRegion(x=x, y=y, width=w, height=h, monitor=monitor_index))

            # Update last frame
            self._last_frames[monitor_index] = current_frame.copy()

        except Exception as e:
            logger.error(f"Change detection error: {e}")

        return changes

    async def shutdown(self) -> None:
        if self._sct:
            self._sct.close()


# =============================================================================
# OCR PROCESSOR
# =============================================================================

class OCRProcessor:
    """Text extraction from screen captures."""

    def __init__(self, config: ScreenReaderConfig):
        self.config = config
        self._engine = config.ocr_engine
        self._languages = config.ocr_languages
        self._threshold = config.ocr_confidence_threshold
        self._reader: Any = None
        self._tesseract_config = "--psm 6 --oem 3"  # Assume single uniform block

    async def initialize(self) -> bool:
        try:
            if self._engine == "tesseract":
                import pytesseract
                # Verify tesseract is available
                pytesseract.get_tesseract_version()
                self._pytesseract = pytesseract
                logger.info("OCRProcessor initialized with Tesseract")
            elif self._engine == "easyocr":
                import easyocr
                self._reader = easyocr.Reader(self._languages, gpu=True)
                logger.info("OCRProcessor initialized with EasyOCR")
            elif self._engine == "paddleocr":
                from paddleocr import PaddleOCR
                self._reader = PaddleOCR(use_angle_cls=True, lang=self._languages[0], use_gpu=True)
                logger.info("OCRProcessor initialized with PaddleOCR")
            else:
                logger.error(f"Unknown OCR engine: {self._engine}")
                return False

            return True
        except ImportError as e:
            logger.error(f"OCR engine not available: {e}")
            return False
        except Exception as e:
            logger.exception(f"OCRProcessor init error: {e}")
            return False

    def extract_text(self, frame: np.ndarray) -> list[UIElement]:
        """Extract text blocks from frame."""
        elements = []

        try:
            if self._engine == "tesseract":
                elements = self._extract_tesseract(frame)
            elif self._engine in ["easyocr", "paddleocr"] and self._reader:
                elements = self._extract_deep(frame)

        except Exception as e:
            logger.error(f"OCR extraction error: {e}")

        return elements

    def _extract_tesseract(self, frame: np.ndarray) -> list[UIElement]:
        """Extract using Tesseract."""
        elements = []

        try:
            data = self._pytesseract.image_to_data(
                frame,
                lang="+".join(self._languages),
                config=self._tesseract_config,
                output_type=self._pytesseract.Output.DICT,
            )

            h, w = frame.shape[:2]
            n_boxes = len(data["text"])

            for i in range(n_boxes):
                conf = int(data["conf"][i])
                if conf > self._threshold * 100:
                    text = data["text"][i].strip()
                    if text and len(text) > 1:
                        x, y, w_box, h_box = (
                            data["left"][i],
                            data["top"][i],
                            data["width"][i],
                            data["height"][i],
                        )

                        # Group nearby text into blocks
                        elements.append(UIElement(
                            element_id=str(uuid.uuid4()),
                            element_type="text",
                            bbox=ScreenRegion(x=x, y=y, width=w_box, height=h_box),
                            text=text,
                            confidence=conf / 100.0,
                            attributes={"block_num": data["block_num"][i], "line_num": data["line_num"][i]},
                        ))

        except Exception as e:
            logger.error(f"Tesseract extraction error: {e}")

        return elements

    def _extract_deep(self, frame: np.ndarray) -> list[UIElement]:
        """Extract using EasyOCR or PaddleOCR."""
        elements = []

        try:
            results = self._reader.readtext(frame)
            h, w = frame.shape[:2]

            for result in results:
                bbox_coords, text, conf = result
                if conf > self._threshold and text.strip():
                    xs = [p[0] for p in bbox_coords]
                    ys = [p[1] for p in bbox_coords]
                    x1, x2 = int(min(xs)), int(max(xs))
                    y1, y2 = int(min(ys)), int(max(ys))

                    elements.append(UIElement(
                        element_id=str(uuid.uuid4()),
                        element_type="text",
                        bbox=ScreenRegion(x=x1, y=y1, width=x2-x1, height=y2-y1),
                        text=text.strip(),
                        confidence=conf,
                    ))

        except Exception as e:
            logger.error(f"Deep OCR extraction error: {e}")

        return elements

    async def shutdown(self) -> None:
        self._reader = None


# =============================================================================
# OMNIPARSER CLIENT (UI Element Detection)
# =============================================================================

class OmniParserClient:
    """Client for Microsoft OmniParser service."""

    def __init__(self, endpoint: str, confidence: float = 0.3):
        self.endpoint = endpoint.rstrip("/")
        self.confidence = confidence
        self._session: Any = None

    async def initialize(self) -> bool:
        try:
            import aiohttp
            self._session = aiohttp.ClientSession()
            # Test connection
            async with self._session.get(f"{self.endpoint}/health", timeout=5) as resp:
                if resp.status == 200:
                    logger.info("OmniParserClient connected")
                    return True
        except Exception as e:
            logger.warning(f"OmniParser not available: {e}")
        return False

    async def parse_screen(self, frame: np.ndarray) -> list[UIElement]:
        """Parse screen for UI elements."""
        if not self._session:
            return []

        try:
            import aiohttp
            import base64
            import io

            # Encode frame as base64 JPEG
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_base64 = base64.b64encode(buffer).decode()

            async with self._session.post(
                f"{self.endpoint}/parse",
                json={"image": img_base64, "confidence": self.confidence},
                timeout=30,
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return self._parse_response(data, frame.shape)

        except Exception as e:
            logger.error(f"OmniParser request error: {e}")

        return []

    def _parse_response(self, data: dict, frame_shape: tuple) -> list[UIElement]:
        """Parse OmniParser response."""
        elements = []
        h, w = frame_shape[:2]

        for item in data.get("elements", []):
            bbox = item.get("bbox", [0, 0, 0, 0])  # x1, y1, x2, y2
            x1, y1, x2, y2 = bbox

            elements.append(UIElement(
                element_id=str(uuid.uuid4()),
                element_type=item.get("type", "unknown"),
                bbox=ScreenRegion(
                    x=int(x1),
                    y=int(y1),
                    width=int(x2 - x1),
                    height=int(y2 - y1),
                ),
                text=item.get("text", ""),
                confidence=item.get("confidence", 0.0),
                attributes=item.get("attributes", {}),
                role=item.get("role", ""),
                name=item.get("name", ""),
                state=item.get("state", ""),
                clickable=item.get("clickable", False),
                focused=item.get("focused", False),
                enabled=item.get("enabled", True),
            ))

        return elements

    async def shutdown(self) -> None:
        if self._session:
            await self._session.close()


# =============================================================================
# WINDOW TRACKER
# =============================================================================

class WindowTracker:
    """Track application windows across platforms."""

    def __init__(self, config: ScreenReaderConfig):
        self.config = config
        self._windows: dict[str, WindowInfo] = {}
        self._platform_backend: Any = None

    async def initialize(self) -> bool:
        """Initialize platform-specific window tracking."""
        import platform
        system = platform.system().lower()

        try:
            if system == "windows":
                await self._init_windows()
            elif system == "darwin":
                await self._init_macos()
            elif system == "linux":
                await self._init_linux()
            else:
                logger.warning(f"Window tracking not supported on {system}")
                return False

            logger.info("WindowTracker initialized")
            return True
        except Exception as e:
            logger.error(f"WindowTracker init error: {e}")
            return False

    async def _init_windows(self) -> None:
        """Windows: Use pywin32 or uiautomation."""
        try:
            import uiautomation as auto
            self._platform_backend = auto
            self._backend_type = "uiautomation"
        except ImportError:
            try:
                import win32gui
                import win32process
                self._platform_backend = (win32gui, win32process)
                self._backend_type = "win32"
            except ImportError:
                logger.warning("No Windows window tracking backend available")

    async def _init_macos(self) -> None:
        """macOS: Use Quartz/AppKit via PyObjC."""
        try:
            import Quartz
            self._platform_backend = Quartz
            self._backend_type = "quartz"
        except ImportError:
            logger.warning("PyObjC not available for macOS window tracking")

    async def _init_linux(self) -> None:
        """Linux: Use Xlib or wmctrl."""
        try:
            import Xlib.display
            import Xlib.X
            self._platform_backend = Xlib.display.Display()
            self._backend_type = "xlib"
        except ImportError:
            try:
                import subprocess
                # Fallback to wmctrl
                subprocess.run(["wmctrl", "-l"], check=True, capture_output=True)
                self._backend_type = "wmctrl"
            except Exception:
                logger.warning("No Linux window tracking backend available")

    def get_windows(self) -> list[WindowInfo]:
        """Get list of all visible windows."""
        windows = []

        if self._backend_type == "uiautomation":
            windows = self._get_windows_uia()
        elif self._backend_type == "win32":
            windows = self._get_windows_win32()
        elif self._backend_type == "quartz":
            windows = self._get_windows_quartz()
        elif self._backend_type == "xlib":
            windows = self._get_windows_xlib()
        elif self._backend_type == "wmctrl":
            windows = self._get_windows_wmctrl()

        self._windows = {w.window_id: w for w in windows}
        return windows

    def _get_windows_uia(self) -> list[WindowInfo]:
        """Get windows using UIAutomation."""
        windows = []
        try:
            auto = self._platform_backend
            root = auto.GetRootControl()
            for win in root.GetChildren():
                try:
                    rect = win.BoundingRectangle
                    windows.append(WindowInfo(
                        window_id=str(win.NativeWindowHandle),
                        title=win.Name or "",
                        app_name=win.ClassName or "",
                        pid=win.ProcessId,
                        bounds=ScreenRegion(
                            x=rect.left, y=rect.top,
                            width=rect.width(), height=rect.height()
                        ),
                        is_active=win.IsActive,
                        is_visible=win.IsVisible,
                        z_order=0,
                    ))
                except Exception:
                    pass
        return windows

    def _get_windows_win32(self) -> list[WindowInfo]:
        """Get windows using win32 API."""
        windows = []
        try:
            win32gui, win32process = self._platform_backend

            def callback(hwnd, _):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if title:
                        _, pid = win32process.GetWindowThreadProcessId(hwnd)
                        rect = win32gui.GetWindowRect(hwnd)
                        windows.append(WindowInfo(
                            window_id=str(hwnd),
                            title=title,
                            app_name=win32gui.GetClassName(hwnd),
                            pid=pid,
                            bounds=ScreenRegion(
                                x=rect[0], y=rect[1],
                                width=rect[2]-rect[0], height=rect[3]-rect[1]
                            ),
                            is_active=(hwnd == win32gui.GetForegroundWindow()),
                            is_visible=True,
                            z_order=0,
                        ))
                return True

            win32gui.EnumWindows(callback, None)
        except Exception as e:
            logger.error(f"Win32 window enum error: {e}")

        return windows

    def _get_windows_quartz(self) -> list[WindowInfo]:
        """Get windows on macOS using Quartz."""
        windows = []
        try:
            Quartz = self._platform_backend
            window_list = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
                Quartz.kCGNullWindowID,
            )

            for win in window_list:
                bounds = win.get(Quartz.kCGWindowBounds, {})
                windows.append(WindowInfo(
                    window_id=str(win.get(Quartz.kCGWindowNumber, 0)),
                    title=win.get(Quartz.kCGWindowName, "") or "",
                    app_name=win.get(Quartz.kCGWindowOwnerName, "") or "",
                    pid=win.get(Quartz.kCGWindowOwnerPID, 0),
                    bounds=ScreenRegion(
                        x=int(bounds.get("X", 0)),
                        y=int(bounds.get("Y", 0)),
                        width=int(bounds.get("Width", 0)),
                        height=int(bounds.get("Height", 0)),
                    ),
                    is_active=win.get(Quartz.kCGWindowLayer, 0) == 0,
                    is_visible=True,
                    z_order=win.get(Quartz.kCGWindowLayer, 0),
                ))
        except Exception as e:
            logger.error(f"Quartz window enum error: {e}")

        return windows

    def _get_windows_xlib(self) -> list[WindowInfo]:
        """Get windows on Linux using Xlib."""
        windows = []
        try:
            display = self._platform_backend
            root = display.screen().root
            window_ids = root.query_tree().children

            for win_id in window_ids:
                try:
                    attrs = win_id.get_attributes()
                    if attrs.map_state == Xlib.X.IsViewable:
                        geom = win_id.get_geometry()
                        name = win_id.get_wm_name() or ""
                        cls = win_id.get_wm_class() or ("", "")
                        windows.append(WindowInfo(
                            window_id=str(win_id),
                            title=name,
                            app_name=cls[1] if cls[1] else cls[0],
                            pid=0,
                            bounds=ScreenRegion(
                                x=geom.x, y=geom.y,
                                width=geom.width, height=geom.height
                            ),
                            is_active=False,
                            is_visible=True,
                            z_order=0,
                        ))
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Xlib window enum error: {e}")

        return windows

    def _get_windows_wmctrl(self) -> list[WindowInfo]:
        """Get windows using wmctrl."""
        windows = []
        try:
            import subprocess
            result = subprocess.run(["wmctrl", "-l", "-G"], capture_output=True, text=True, check=True)
            for line in result.stdout.strip().split("\n"):
                parts = line.split(None, 6)
                if len(parts) >= 7:
                    win_id, desktop, x, y, w, h, title = parts
                    windows.append(WindowInfo(
                        window_id=win_id,
                        title=title,
                        app_name="",
                        pid=0,
                        bounds=ScreenRegion(
                            x=int(x), y=int(y), width=int(w), height=int(h)
                        ),
                        is_active=False,
                        is_visible=True,
                        z_order=int(desktop),
                    ))
        except Exception as e:
            logger.error(f"wmctrl error: {e}")

        return windows

    def get_active_window(self) -> WindowInfo | None:
        """Get currently active window."""
        for win in self.get_windows():
            if win.is_active:
                return win
        return None


# =============================================================================
# SCREEN READER (Main Orchestrator)
# =============================================================================

class ScreenReader:
    """
    High-level screen reader orchestrating capture, OCR, UI parsing, and monitoring.
    """

    def __init__(
        self,
        websocket_manager: Any = None,
        event_bus: Any = None,
        config: ScreenReaderConfig | None = None,
    ):
        self.websocket_manager = websocket_manager
        self.event_bus = event_bus
        self.config = config or ScreenReaderConfig()

        # Components
        self.capturer = ScreenCapturer(self.config)
        self.ocr_processor = OCRProcessor(self.config) if config.ocr_enabled else None
        self.omniparser = OmniParserClient(
            config.omniparser_endpoint,
            config.omniparser_confidence,
        ) if config.omniparser_enabled else None
        self.window_tracker = WindowTracker(self.config) if config.track_windows else None

        # State
        self._running = False
        self._capture_task: asyncio.Task | None = None
        self._analysis_task: asyncio.Task | None = None
        self._window_task: asyncio.Task | None = None
        self._frame_count = 0

        # Callbacks
        self._on_capture: Callable[[ScreenCaptureResult], Any] | None = None
        self._on_text_found: Callable[[list[UIElement]], Any] | None = None
        self._on_ui_elements: Callable[[list[UIElement]], Any] | None = None
        self._on_change_detected: Callable[[list[ScreenRegion]], Any] | None = None
        self._on_window_change: Callable[[list[WindowInfo]], Any] | None = None

        # Stats
        self._stats = {
            "captures": 0,
            "ocr_runs": 0,
            "ui_parses": 0,
            "changes_detected": 0,
            "avg_capture_ms": 0.0,
        }

    async def start(self) -> bool:
        """Start screen reader."""
        if self._running:
            return True

        # Initialize components
        await self.capturer.initialize()

        if self.ocr_processor:
            await self.ocr_processor.initialize()

        if self.omniparser:
            await self.omniparser.initialize()

        if self.window_tracker:
            await self.window_tracker.initialize()

        # Start tasks
        self._running = True
        self._capture_task = asyncio.create_task(self._capture_loop())
        self._analysis_task = asyncio.create_task(self._analysis_loop())
        if self.window_tracker:
            self._window_task = asyncio.create_task(self._window_loop())

        logger.info("ScreenReader started")
        return True

    async def stop(self) -> None:
        """Stop screen reader."""
        self._running = False

        for task in [self._capture_task, self._analysis_task, self._window_task]:
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        await self.capturer.shutdown()
        if self.ocr_processor:
            await self.ocr_processor.shutdown()
        if self.omniparser:
            await self.omniparser.shutdown()

        logger.info("ScreenReader stopped")

    async def _capture_loop(self) -> None:
        """Continuous screen capture."""
        interval = 1.0 / self.config.capture_fps

        while self._running:
            start = time.time()

            try:
                frames = self.capturer.capture_all_monitors()

                for monitor_idx, frame in frames.items():
                    result = ScreenCaptureResult(
                        capture_id=str(uuid.uuid4()),
                        timestamp=time.time(),
                        monitor=monitor_idx,
                        region=None,
                        frame=frame,
                        width=frame.shape[1],
                        height=frame.shape[0],
                    )

                    # Queue for analysis
                    self._frame_count += 1

                    # Callbacks
                    if self._on_capture:
                        await self._on_capture(result)

                    # Broadcast to HUD
                    if self.websocket_manager:
                        await self.websocket_manager.broadcast({
                            "type": "screen_capture",
                            "payload": {
                                "monitor": monitor_idx,
                                "width": frame.shape[1],
                                "height": frame.shape[0],
                                "timestamp": result.timestamp,
                            },
                        }, room="hud")

                self._stats["captures"] += len(frames)

            except Exception as e:
                logger.error(f"Capture loop error: {e}")

            elapsed = time.time() - start
            await asyncio.sleep(max(0, interval - elapsed))

    async def _analysis_loop(self) -> None:
        """Periodic analysis (OCR, UI parsing)."""
        while self._running:
            try:
                # This would process frames from a queue
                # For now, simplified - runs on capture frames
                await asyncio.sleep(self.config.ocr_interval / self.config.capture_fps)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Analysis loop error: {e}")

    async def _window_loop(self) -> None:
        """Periodic window tracking."""
        while self._running:
            try:
                windows = self.window_tracker.get_windows()
                if self._on_window_change:
                    await self._on_window_change(windows)
            except Exception as e:
                logger.error(f"Window loop error: {e}")
            await asyncio.sleep(self.config.window_update_interval)

    # Public API
    def capture_once(self, monitor: int = 0) -> ScreenCaptureResult | None:
        """Capture single frame."""
        frame = self.capturer.capture_monitor(monitor)
        if frame is None:
            return None

        return ScreenCaptureResult(
            capture_id=str(uuid.uuid4()),
            timestamp=time.time(),
            monitor=monitor,
            region=None,
            frame=frame,
            width=frame.shape[1],
            height=frame.shape[0],
        )

    def capture_region(self, region: ScreenRegion) -> ScreenCaptureResult | None:
        """Capture specific region."""
        frame = self.capturer.capture_region(region)
        if frame is None:
            return None

        return ScreenCaptureResult(
            capture_id=str(uuid.uuid4()),
            timestamp=time.time(),
            monitor=region.monitor,
            region=region,
            frame=frame,
            width=frame.shape[1],
            height=frame.shape[0],
        )

    def extract_text(self, frame: np.ndarray) -> list[UIElement]:
        """Extract text from frame."""
        if self.ocr_processor:
            return self.ocr_processor.extract_text(frame)
        return []

    async def parse_ui(self, frame: np.ndarray) -> list[UIElement]:
        """Parse UI elements using OmniParser."""
        if self.omniparser:
            return await self.omniparser.parse_screen(frame)
        return []

    def get_windows(self) -> list[WindowInfo]:
        """Get current window list."""
        if self.window_tracker:
            return self.window_tracker.get_windows()
        return []

    def get_active_window(self) -> WindowInfo | None:
        """Get active window."""
        if self.window_tracker:
            return self.window_tracker.get_active_window()
        return None

    def get_stats(self) -> dict[str, Any]:
        return self._stats.copy()

    # Callbacks
    def on_capture(self, callback: Callable[[ScreenCaptureResult], Any]) -> None:
        self._on_capture = callback

    def on_text_found(self, callback: Callable[[list[UIElement]], Any]) -> None:
        self._on_text_found = callback

    def on_ui_elements(self, callback: Callable[[list[UIElement]], Any]) -> None:
        self._on_ui_elements = callback

    def on_change_detected(self, callback: Callable[[list[ScreenRegion]], Any]) -> None:
        self._on_change_detected = callback

    def on_window_change(self, callback: Callable[[list[WindowInfo]], Any]) -> None:
        self._on_window_change = callback


# =============================================================================
# FACTORY
# =============================================================================

async def create_screen_reader(
    websocket_manager: Any = None,
    event_bus: Any = None,
    config: ScreenReaderConfig | None = None,
) -> ScreenReader:
    """Factory to create and start screen reader."""
    reader = ScreenReader(
        websocket_manager=websocket_manager,
        event_bus=event_bus,
        config=config,
    )
    await reader.start()
    return reader


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ScreenReader",
    "ScreenCapturer",
    "OCRProcessor",
    "OmniParserClient",
    "WindowTracker",
    "ScreenReaderConfig",
    "ScreenRegion",
    "ScreenRegionType",
    "UIElement",
    "ScreenCaptureResult",
    "WindowInfo",
    "create_screen_reader",
]