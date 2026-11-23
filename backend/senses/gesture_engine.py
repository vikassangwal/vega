"""
JARVIS OMEGA - Gesture Engine
Real-time hand tracking and gesture recognition using MediaPipe.
Supports static gestures, dynamic gestures, and two-hand interactions.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import cv2
import numpy as np

from config import get_settings, settings

logger = logging.getLogger("jarvis.senses.gesture")


# =============================================================================
# DATA MODELS
# =============================================================================

class Handedness(str, Enum):
    LEFT = "Left"
    RIGHT = "Right"


class GestureType(str, Enum):
    # Static gestures
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    PEACE = "peace"              # Victory sign
    FIST = "fist"
    OPEN_PALM = "open_palm"
    POINT = "point"              # Index finger extended
    OK = "ok"                    # Thumb + index circle
    ROCK = "rock"                # Index + pinky
    CALL_ME = "call_me"          # Thumb + pinky
    CROSS_FINGERS = "cross_fingers"
    # Dynamic gestures
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"
    PINCH = "pinch"              # Thumb + index pinch
    GRAB = "grab"                # All fingers close
    RELEASE = "release"          # Open from grab
    ROTATE_CW = "rotate_cw"      # Two-hand rotate clockwise
    ROTATE_CCW = "rotate_ccw"    # Two-hand rotate counter-clockwise
    ZOOM_IN = "zoom_in"          # Two-hand spread
    ZOOM_OUT = "zoom_out"        # Two-hand pinch
    WAVE = "wave"                # Side-to-side
    CIRCLE = "circle"            # Circular motion
    # Special
    NONE = "none"
    UNKNOWN = "unknown"


@dataclass
class HandLandmark:
    """Single hand landmark point."""
    x: float
    y: float
    z: float
    visibility: float = 1.0


@dataclass
class HandData:
    """Complete hand tracking data."""
    handedness: Handedness
    landmarks: list[HandLandmark]  # 21 landmarks
    bbox: tuple[float, float, float, float]  # x, y, w, h normalized
    confidence: float
    # Derived
    center: tuple[float, float] = (0, 0)
    orientation: float = 0.0  # Radians


@dataclass
class GestureEvent:
    """Detected gesture event."""
    gesture: GestureType
    hand: Handedness
    confidence: float
    timestamp: float
    # For dynamic gestures
    start_position: tuple[float, float] | None = None
    end_position: tuple[float, float] | None = None
    velocity: float = 0.0
    direction: float = 0.0  # Radians
    # Metadata
    hand_data: HandData | None = None
    gesture_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class GestureConfig:
    """Gesture engine configuration."""
    # MediaPipe Hands
    max_num_hands: int = 2
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5
    model_complexity: int = 1  # 0=light, 1=full

    # Static gesture thresholds
    static_gesture_confidence: float = 0.8
    finger_extended_threshold: float = 0.15  # Distance ratio
    finger_curled_threshold: float = 0.1

    # Dynamic gesture
    dynamic_enabled: bool = True
    min_swipe_distance: float = 0.15  # Normalized screen distance
    min_swipe_velocity: float = 0.5   # Distance per second
    swipe_time_window: float = 0.5    # Seconds
    pinch_threshold: float = 0.04     # Normalized distance
    grab_threshold: float = 0.08

    # History buffer
    history_size: int = 30  # Frames
    smoothing: float = 0.3  # Landmark smoothing factor

    # Cooldown
    gesture_cooldown: float = 0.5  # Seconds between same gesture

    # Action mapping
    gesture_actions: dict[str, str] = field(default_factory=lambda: {
        "thumbs_up": "confirm",
        "thumbs_down": "cancel",
        "peace": "screenshot",
        "fist": "stop",
        "open_palm": "wake",
        "point": "click",
        "ok": "accept",
        "swipe_left": "prev",
        "swipe_right": "next",
        "swipe_up": "volume_up",
        "swipe_down": "volume_down",
        "pinch": "zoom",
        "grab": "drag_start",
        "release": "drag_end",
        "rotate_cw": "rotate_right",
        "rotate_ccw": "rotate_left",
        "zoom_in": "zoom_in",
        "zoom_out": "zoom_out",
        "wave": "hello",
    })


# =============================================================================
# HAND TRACKER (MediaPipe wrapper)
# =============================================================================

class HandTracker:
    """MediaPipe Hands wrapper with smoothing and utilities."""

    def __init__(self, config: GestureConfig):
        self.config = config
        self._hands: Any = None
        self._mp_hands: Any = None
        self._mp_drawing: Any = None
        self._initialized = False

        # Landmark indices (MediaPipe Hands)
        self.LANDMARKS = {
            "wrist": 0,
            "thumb_cmc": 1, "thumb_mcp": 2, "thumb_ip": 3, "thumb_tip": 4,
            "index_mcp": 5, "index_pip": 6, "index_dip": 7, "index_tip": 8,
            "middle_mcp": 9, "middle_pip": 10, "middle_dip": 11, "middle_tip": 12,
            "ring_mcp": 13, "ring_pip": 14, "ring_dip": 15, "ring_tip": 16,
            "pinky_mcp": 17, "pinky_pip": 18, "pinky_dip": 19, "pinky_tip": 20,
        }

        # Finger tip and pip indices for extension detection
        self.FINGER_TIPS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
        self.FINGER_PIPS = [3, 6, 10, 14, 18]  # corresponding pip joints

    async def initialize(self) -> bool:
        try:
            import mediapipe as mp
            self._mp_hands = mp.solutions.hands
            self._mp_drawing = mp.solutions.drawing_utils

            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.config.max_num_hands,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
                model_complexity=self.config.model_complexity,
            )

            self._initialized = True
            logger.info("HandTracker initialized")
            return True
        except ImportError:
            logger.error("mediapipe not installed. Install: pip install mediapipe")
            return False
        except Exception as e:
            logger.exception(f"HandTracker init error: {e}")
            return False

    def process(self, frame: np.ndarray) -> list[HandData]:
        """Process frame and return detected hands."""
        if not self._initialized or self._hands is None:
            return []

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        results = self._hands.process(rgb_frame)

        hands_data = []
        if results.multi_hand_landmarks:
            for i, (hand_landmarks, handedness) in enumerate(zip(
                results.multi_hand_landmarks,
                results.multi_handedness,
            )):
                hand_label = handedness.classification[0].label
                hand_confidence = handedness.classification[0].score

                # Extract landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append(HandLandmark(
                        x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility
                    ))

                # Calculate bounding box
                xs = [lm.x for lm in landmarks]
                ys = [lm.y for lm in landmarks]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)

                # Center point
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Orientation (wrist to middle MCP)
                wrist = landmarks[self.LANDMARKS["wrist"]]
                middle_mcp = landmarks[self.LANDMARKS["middle_mcp"]]
                orientation = math.atan2(
                    middle_mcp.y - wrist.y,
                    middle_mcp.x - wrist.x,
                )

                hands_data.append(HandData(
                    handedness=Handedness(hand_label),
                    landmarks=landmarks,
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    confidence=hand_confidence,
                    center=(center_x, center_y),
                    orientation=orientation,
                ))

        return hands_data

    def draw_landmarks(self, frame: np.ndarray, hands: list[HandData]) -> np.ndarray:
        """Draw hand landmarks on frame."""
        if not self._initialized or not self._mp_drawing:
            return frame

        # This would require converting HandData back to MediaPipe format
        # Simplified - just draw points
        for hand in hands:
            for lm in hand.landmarks:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            # Draw connections
            connections = self._mp_hands.HAND_CONNECTIONS
            for conn in connections:
                start_idx, end_idx = conn
                if start_idx < len(hand.landmarks) and end_idx < len(hand.landmarks):
                    start = hand.landmarks[start_idx]
                    end = hand.landmarks[end_idx]
                    x1 = int(start.x * frame.shape[1])
                    y1 = int(start.y * frame.shape[0])
                    x2 = int(end.x * frame.shape[1])
                    y2 = int(end.y * frame.shape[0])
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return frame

    async def shutdown(self) -> None:
        if self._hands:
            self._hands.close()
        self._initialized = False


# =============================================================================
# GESTURE RECOGNIZER
# =============================================================================

class GestureRecognizer:
    """
    Recognizes static and dynamic gestures from hand tracking data.
    """

    def __init__(self, config: GestureConfig):
        self.config = config

        # History buffers per hand
        self._landmark_history: dict[Handedness, deque] = {
            Handedness.LEFT: deque(maxlen=config.history_size),
            Handedness.RIGHT: deque(maxlen=config.history_size),
        }
        self._gesture_history: dict[Handedness, deque] = {
            Handedness.LEFT: deque(maxlen=10),
            Handedness.RIGHT: deque(maxlen=10),
        }

        # Last gesture times for cooldown
        self._last_gesture_time: dict[tuple[Handedness, GestureType], float] = {}

        # Smoothing
        self._smoothed_landmarks: dict[Handedness, list[HandLandmark] | None] = {
            Handedness.LEFT: None,
            Handedness.RIGHT: None,
        }

    def update(self, hand_data: HandData) -> list[GestureEvent]:
        """Update with new hand data and return detected gestures."""
        gestures = []
        handedness = hand_data.handedness

        # Smooth landmarks
        smoothed = self._smooth_landmarks(hand_data.landmarks, handedness)
        hand_data.landmarks = smoothed

        # Add to history
        self._landmark_history[handedness].append({
            "landmarks": smoothed,
            "center": hand_data.center,
            "timestamp": time.time(),
        })

        # Detect static gesture
        static_gesture = self._recognize_static(smoothed)
        if static_gesture != GestureType.NONE:
            if self._check_cooldown(handedness, static_gesture):
                gestures.append(GestureEvent(
                    gesture=static_gesture,
                    hand=handedness,
                    confidence=self.config.static_gesture_confidence,
                    timestamp=time.time(),
                    hand_data=hand_data,
                ))
                self._last_gesture_time[(handedness, static_gesture)] = time.time()
                self._gesture_history[handedness].append(static_gesture)

        # Detect dynamic gestures
        if self.config.dynamic_enabled:
            dynamic_gestures = self._recognize_dynamic(handedness)
            for g in dynamic_gestures:
                if self._check_cooldown(handedness, g.gesture):
                    gestures.append(g)
                    self._last_gesture_time[(handedness, g.gesture)] = time.time()

        return gestures

    def _smooth_landmarks(
        self,
        landmarks: list[HandLandmark],
        handedness: Handedness,
    ) -> list[HandLandmark]:
        """Apply exponential smoothing to landmarks."""
        prev = self._smoothed_landmarks[handedness]
        alpha = self.config.smoothing

        if prev is None:
            self._smoothed_landmarks[handedness] = landmarks
            return landmarks

        smoothed = []
        for i, (curr, prev_lm) in enumerate(zip(landmarks, prev)):
            smoothed.append(HandLandmark(
                x=alpha * curr.x + (1 - alpha) * prev_lm.x,
                y=alpha * curr.y + (1 - alpha) * prev_lm.y,
                z=alpha * curr.z + (1 - alpha) * prev_lm.z,
                visibility=curr.visibility,
            ))

        self._smoothed_landmarks[handedness] = smoothed
        return smoothed

    def _recognize_static(self, landmarks: list[HandLandmark]) -> GestureType:
        """Recognize static hand gesture."""
        # Get finger states
        fingers_extended = self._get_finger_states(landmarks)

        # Thumb, Index, Middle, Ring, Pinky
        thumb, index, middle, ring, pinky = fingers_extended

        # Gesture classification
        if thumb and not index and not middle and not ring and not pinky:
            return GestureType.THUMBS_UP
        if not thumb and not index and not middle and not ring and not pinky:
            return GestureType.FIST
        if not thumb and index and middle and not ring and not pinky:
            return GestureType.PEACE
        if thumb and index and middle and ring and pinky:
            return GestureType.OPEN_PALM
        if not thumb and index and not middle and not ring and not pinky:
            return GestureType.POINT
        if thumb and index and not middle and not ring and pinky:
            return GestureType.ROCK
        if thumb and not index and not middle and not ring and pinky:
            return GestureType.CALL_ME
        # OK sign - thumb and index tips close
        if self._fingers_touching(landmarks, 4, 8):  # thumb_tip, index_tip
            return GestureType.OK
        # Crossed fingers
        if index and middle and self._fingers_crossed(landmarks, 8, 12):
            return GestureType.CROSS_FINGERS

        return GestureType.NONE

    def _get_finger_states(self, landmarks: list[HandLandmark]) -> list[bool]:
        """Determine which fingers are extended."""
        states = []

        # Thumb: compare tip to IP joint (x-axis for horizontal)
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]

        # Thumb extended if tip is further from wrist than IP
        wrist = landmarks[0]
        thumb_dist = math.hypot(thumb_tip.x - wrist.x, thumb_tip.y - wrist.y)
        ip_dist = math.hypot(thumb_ip.x - wrist.x, thumb_ip.y - wrist.y)
        states.append(thumb_dist > ip_dist * 1.2)

        # Other fingers: tip vs PIP (y-axis, tip higher = extended)
        for tip_idx, pip_idx in zip([8, 12, 16, 20], [6, 10, 14, 18]):
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            # In image coordinates, y increases downward
            # Extended = tip.y < pip.y (tip is higher up)
            extended = (pip.y - tip.y) > self.config.finger_extended_threshold
            states.append(extended)

        return states

    def _fingers_touching(self, landmarks: list[HandLandmark], idx1: int, idx2: int) -> bool:
        """Check if two fingertips are close."""
        lm1 = landmarks[idx1]
        lm2 = landmarks[idx2]
        dist = math.hypot(lm1.x - lm2.x, lm1.y - lm2.y)
        return dist < self.config.pinch_threshold

    def _fingers_crossed(self, landmarks: list[HandLandmark], idx1: int, idx2: int) -> bool:
        """Check if two fingers are crossed (X shape)."""
        lm1 = landmarks[idx1]
        lm2 = landmarks[idx2]
        dx = abs(lm1.x - lm2.x)
        dy = abs(lm1.y - lm2.y)
        return dx < 0.05 and dy > 0.05

    def _recognize_dynamic(self, handedness: Handedness) -> list[GestureEvent]:
        """Recognize dynamic gestures from history."""
        gestures = []
        history = list(self._landmark_history[handedness])

        if len(history) < 5:
            return gestures

        # Get recent centers
        centers = [h["center"] for h in history[-10:]]
        timestamps = [h["timestamp"] for h in history[-10:]]

        # Swipe detection
        swipe = self._detect_swipe(centers, timestamps)
        if swipe:
            gestures.append(swipe)

        # Pinch detection (thumb + index distance)
        pinch = self._detect_pinch(history)
        if pinch:
            gestures.append(pinch)

        # Grab/Release
        grab = self._detect_grab(history)
        if grab:
            gestures.append(grab)

        return gestures

    def _detect_swipe(
        self,
        centers: list[tuple[float, float]],
        timestamps: list[float],
    ) -> GestureEvent | None:
        """Detect swipe gesture from center trajectory."""
        if len(centers) < 5:
            return None

        # Calculate total displacement
        start = centers[0]
        end = centers[-1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.hypot(dx, dy)

        if distance < self.config.min_swipe_distance:
            return None

        # Calculate velocity
        dt = timestamps[-1] - timestamps[0]
        if dt <= 0:
            return None
        velocity = distance / dt

        if velocity < self.config.min_swipe_velocity:
            return None

        # Determine direction
        angle = math.atan2(dy, dx)  # -pi to pi

        if -math.pi/4 <= angle < math.pi/4:
            gesture = GestureType.SWIPE_RIGHT
        elif math.pi/4 <= angle < 3*math.pi/4:
            gesture = GestureType.SWIPE_DOWN
        elif -3*math.pi/4 <= angle < -math.pi/4:
            gesture = GestureType.SWIPE_UP
        else:
            gesture = GestureType.SWIPE_LEFT

        return GestureEvent(
            gesture=gesture,
            hand=Handedness.RIGHT,  # Will be set by caller
            confidence=min(1.0, velocity / 2.0),
            timestamp=time.time(),
            start_position=start,
            end_position=end,
            velocity=velocity,
            direction=angle,
        )

    def _detect_pinch(self, history: list[dict]) -> GestureEvent | None:
        """Detect pinch gesture (thumb + index)."""
        if len(history) < 3:
            return None

        # Check recent frames for pinch
        pinching = 0
        for h in history[-5:]:
            landmarks = h["landmarks"]
            if len(landmarks) > 8:
                dist = math.hypot(
                    landmarks[4].x - landmarks[8].x,
                    landmarks[4].y - landmarks[8].y,
                )
                if dist < self.config.pinch_threshold:
                    pinching += 1

        if pinching >= 3:
            return GestureEvent(
                gesture=GestureType.PINCH,
                hand=Handedness.RIGHT,
                confidence=0.8,
                timestamp=time.time(),
            )

        return None

    def _detect_grab(self, history: list[dict]) -> GestureEvent | None:
        """Detect grab/release gesture."""
        if len(history) < 3:
            return None

        # Check if all fingertips close to palm
        grabbing = 0
        releasing = 0

        for h in history[-5:]:
            landmarks = h["landmarks"]
            if len(landmarks) < 21:
                continue

            wrist = landmarks[0]
            tips = [landmarks[i] for i in [4, 8, 12, 16, 20]]

            avg_dist = sum(math.hypot(tip.x - wrist.x, tip.y - wrist.y) for tip in tips) / 5

            if avg_dist < self.config.grab_threshold:
                grabbing += 1
            else:
                releasing += 1

        if grabbing >= 3:
            return GestureEvent(
                gesture=GestureType.GRAB,
                hand=Handedness.RIGHT,
                confidence=0.8,
                timestamp=time.time(),
            )
        elif releasing >= 3:
            return GestureEvent(
                gesture=GestureType.RELEASE,
                hand=Handedness.RIGHT,
                confidence=0.8,
                timestamp=time.time(),
            )

        return None

    def _check_cooldown(self, handedness: Handedness, gesture: GestureType) -> bool:
        """Check if gesture is off cooldown."""
        key = (handedness, gesture)
        last_time = self._last_gesture_time.get(key, 0)
        return time.time() - last_time > self.config.gesture_cooldown


# =============================================================================
# TWO-HAND GESTURE RECOGNIZER
# =============================================================================

class TwoHandGestureRecognizer:
    """Recognizes gestures requiring both hands."""

    def __init__(self, config: GestureConfig):
        self.config = config
        self._left_history: deque = deque(maxlen=config.history_size)
        self._right_history: deque = deque(maxlen=config.history_size)

    def update(self, left_hand: HandData | None, right_hand: HandData | None) -> list[GestureEvent]:
        """Update with both hands and return two-hand gestures."""
        gestures = []

        if left_hand:
            self._left_history.append({
                "center": left_hand.center,
                "landmarks": left_hand.landmarks,
                "timestamp": time.time(),
            })
        if right_hand:
            self._right_history.append({
                "center": right_hand.center,
                "landmarks": right_hand.landmarks,
                "timestamp": time.time(),
            })

        if len(self._left_history) < 5 or len(self._right_history) < 5:
            return gestures

        # Zoom (pinch/spread with both hands)
        zoom = self._detect_zoom()
        if zoom:
            gestures.append(zoom)

        # Rotate
        rotate = self._detect_rotate()
        if rotate:
            gestures.append(rotate)

        return gestures

    def _detect_zoom(self) -> GestureEvent | None:
        """Detect two-hand zoom in/out."""
        left_centers = [h["center"] for h in list(self._left_history)[-5:]]
        right_centers = [h["center"] for h in list(self._right_history)[-5:]]

        if len(left_centers) < 3 or len(right_centers) < 3:
            return None

        # Distance between hands
        start_dist = math.hypot(
            left_centers[0][0] - right_centers[0][0],
            left_centers[0][1] - right_centers[0][1],
        )
        end_dist = math.hypot(
            left_centers[-1][0] - right_centers[-1][0],
            left_centers[-1][1] - right_centers[-1][1],
        )

        diff = end_dist - start_dist
        threshold = 0.05  # 5% screen change

        if abs(diff) > threshold:
            if diff > 0:
                return GestureEvent(
                    gesture=GestureType.ZOOM_IN,
                    hand=Handedness.LEFT,
                    confidence=min(1.0, abs(diff) * 10),
                    timestamp=time.time(),
                )
            else:
                return GestureEvent(
                    gesture=GestureType.ZOOM_OUT,
                    hand=Handedness.LEFT,
                    confidence=min(1.0, abs(diff) * 10),
                    timestamp=time.time(),
                )

        return None

    def _detect_rotate(self) -> GestureEvent | None:
        """Detect two-hand rotation."""
        left_centers = [h["center"] for h in list(self._left_history)[-5:]]
        right_centers = [h["center"] for h in list(self._right_history)[-5:]]

        if len(left_centers) < 3 or len(right_centers) < 3:
            return None

        # Angle between hands
        def angle(c1, c2):
            return math.atan2(c2[1] - c1[1], c2[0] - c1[0])

        start_angle = angle(left_centers[0], right_centers[0])
        end_angle = angle(left_centers[-1], right_centers[-1])

        # Normalize angle difference
        diff = end_angle - start_angle
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi

        threshold = 0.3  # ~17 degrees

        if abs(diff) > threshold:
            if diff > 0:
                return GestureEvent(
                    gesture=GestureType.ROTATE_CCW,
                    hand=Handedness.LEFT,
                    confidence=min(1.0, abs(diff) * 2),
                    timestamp=time.time(),
                )
            else:
                return GestureEvent(
                    gesture=GestureType.ROTATE_CW,
                    hand=Handedness.LEFT,
                    confidence=min(1.0, abs(diff) * 2),
                    timestamp=time.time(),
                )

        return None


# =============================================================================
# GESTURE ENGINE (Main Orchestrator)
# =============================================================================

class GestureEngine:
    """
    High-level gesture engine managing hand tracking, recognition, and action mapping.
    """

    def __init__(
        self,
        websocket_manager: Any = None,
        event_bus: Any = None,
        config: GestureConfig | None = None,
    ):
        self.websocket_manager = websocket_manager
        self.event_bus = event_bus
        self.config = config or GestureConfig()

        # Components
        self.hand_tracker = HandTracker(self.config)
        self.gesture_recognizer = GestureRecognizer(self.config)
        self.two_hand_recognizer = TwoHandGestureRecognizer(self.config)

        # State
        self._running = False
        self._process_task: asyncio.Task | None = None
        self._frame_queue: asyncio.Queue = asyncio.Queue(maxsize=5)

        # Latest hands
        self._left_hand: HandData | None = None
        self._right_hand: HandData | None = None

        # Callbacks
        self._on_gesture: Callable[[GestureEvent], Any] | None = None
        self._on_hands: Callable[[list[HandData]], Any] | None = None

        # Action cooldowns
        self._action_cooldowns: dict[str, float] = {}

        # Stats
        self._stats = {
            "frames_processed": 0,
            "hands_detected": 0,
            "gestures_recognized": 0,
            "actions_triggered": 0,
        }

    async def start(self) -> bool:
        """Start gesture engine."""
        if self._running:
            return True

        # Initialize hand tracker
        if not await self.hand_tracker.initialize():
            logger.error("HandTracker failed to initialize")
            return False

        self._running = True
        self._process_task = asyncio.create_task(self._process_loop())

        logger.info("GestureEngine started")
        return True

    async def stop(self) -> None:
        """Stop gesture engine."""
        self._running = False

        if self._process_task:
            self._process_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._process_task

        await self.hand_tracker.shutdown()
        logger.info("GestureEngine stopped")

    async def _process_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                # Get frame from queue (fed by vision engine or webcam)
                try:
                    frame = await asyncio.wait_for(self._frame_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                # Process hands
                hands = self.hand_tracker.process(frame)

                # Separate left/right
                self._left_hand = None
                self._right_hand = None
                for hand in hands:
                    if hand.handedness == Handedness.LEFT:
                        self._left_hand = hand
                    else:
                        self._right_hand = hand

                # Recognize single-hand gestures
                all_gestures = []
                for hand in hands:
                    gestures = self.gesture_recognizer.update(hand)
                    all_gestures.extend(gestures)

                # Recognize two-hand gestures
                two_hand_gestures = self.two_hand_recognizer.update(
                    self._left_hand, self._right_hand
                )
                all_gestures.extend(two_hand_gestures)

                # Process gestures
                for gesture in all_gestures:
                    await self._handle_gesture(gesture)

                # Update stats
                self._stats["frames_processed"] += 1
                self._stats["hands_detected"] += len(hands)
                self._stats["gestures_recognized"] += len(all_gestures)

                # Callback
                if self._on_hands:
                    await self._on_hands(hands)

                # Broadcast to HUD
                if self.websocket_manager and hands:
                    await self.websocket_manager.broadcast({
                        "type": "hands_detected",
                        "payload": {
                            "count": len(hands),
                            "hands": [
                                {
                                    "handedness": h.handedness.value,
                                    "center": h.center,
                                    "confidence": h.confidence,
                                }
                                for h in hands
                            ],
                        },
                    }, room="hud")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Gesture process loop error: {e}")
                await asyncio.sleep(0.1)

    async def _handle_gesture(self, gesture: GestureEvent) -> None:
        """Handle recognized gesture - map to action."""
        action = self.config.gesture_actions.get(gesture.gesture.value)

        if action:
            # Check action cooldown
            last_time = self._action_cooldowns.get(action, 0)
            if time.time() - last_time < 1.0:  # 1 second action cooldown
                return

            self._action_cooldowns[action] = time.time()
            self._stats["actions_triggered"] += 1

            logger.info(f"Gesture action: {gesture.gesture.value} -> {action}")

            # Emit event
            if self.event_bus:
                await self.event_bus.emit("gesture.action", {
                    "gesture": gesture.gesture.value,
                    "action": action,
                    "hand": gesture.hand.value,
                    "confidence": gesture.confidence,
                })

            # Broadcast to HUD
            if self.websocket_manager:
                await self.websocket_manager.broadcast({
                    "type": "gesture_action",
                    "payload": {
                        "gesture": gesture.gesture.value,
                        "action": action,
                        "hand": gesture.hand.value,
                        "confidence": gesture.confidence,
                    },
                }, room="hud")

        # Callback
        if self._on_gesture:
            await self._on_gesture(gesture)

    def feed_frame(self, frame: np.ndarray) -> bool:
        """Feed frame for processing (non-blocking)."""
        try:
            self._frame_queue.put_nowait(frame)
            return True
        except asyncio.QueueFull:
            return False

    def get_latest_hands(self) -> tuple[HandData | None, HandData | None]:
        """Get latest detected hands."""
        return self._left_hand, self._right_hand

    def get_stats(self) -> dict[str, Any]:
        return self._stats.copy()

    def set_gesture_action(self, gesture: GestureType, action: str) -> None:
        """Map gesture to action."""
        self.config.gesture_actions[gesture.value] = action

    def remove_gesture_action(self, gesture: GestureType) -> None:
        """Remove gesture action mapping."""
        self.config.gesture_actions.pop(gesture.value, None)

    # Callbacks
    def on_gesture(self, callback: Callable[[GestureEvent], Any]) -> None:
        self._on_gesture = callback

    def on_hands(self, callback: Callable[[list[HandData]], Any]) -> None:
        self._on_hands = callback


# =============================================================================
# FACTORY
# =============================================================================

async def create_gesture_engine(
    websocket_manager: Any = None,
    event_bus: Any = None,
    config: GestureConfig | None = None,
) -> GestureEngine:
    """Factory to create and start gesture engine."""
    engine = GestureEngine(
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
    "GestureEngine",
    "HandTracker",
    "GestureRecognizer",
    "TwoHandGestureRecognizer",
    "GestureConfig",
    "Handedness",
    "GestureType",
    "HandLandmark",
    "HandData",
    "GestureEvent",
    "create_gesture_engine",
]