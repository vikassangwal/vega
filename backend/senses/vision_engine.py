"""
JARVIS OMEGA - Vision Engine
Multi-source visual perception: YOLOv8 object detection, DeepFace/InsightFace face recognition,
MediaPipe hands/pose/face mesh, OmniParser screen UI parsing.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Optional

import cv2
import numpy as np

from config import get_settings, settings

logger = logging.getLogger("jarvis.senses.vision")


# =============================================================================
# DATA MODELS
# =============================================================================

class VisionSource(str, Enum):
    WEBCAM = "webcam"
    SCREEN = "screen"
    IMAGE = "image"
    VIDEO = "video"


class DetectionType(str, Enum):
    OBJECT = "object"
    FACE = "face"
    POSE = "pose"
    HAND = "hand"
    GESTURE = "gesture"
    SCREEN_UI = "screen_ui"
    TEXT = "text"
    EMOTION = "emotion"


@dataclass
class BoundingBox:
    """Normalized bounding box [0,1]."""
    x: float
    y: float
    width: float
    height: float

    @property
    def x1(self) -> float: return self.x
    @property
    def y1(self) -> float: return self.y
    @property
    def x2(self) -> float: return self.x + self.width
    @property
    def y2(self) -> float: return self.y + self.height

    def to_pixel(self, img_width: int, img_height: int) -> tuple[int, int, int, int]:
        return (
            int(self.x * img_width),
            int(self.y * img_height),
            int(self.width * img_width),
            int(self.height * img_height),
        )

    @classmethod
    def from_pixel(cls, x: int, y: int, w: int, h: int, img_width: int, img_height: int) -> "BoundingBox":
        return cls(x / img_width, y / img_height, w / img_width, h / img_height)


@dataclass
class Detection:
    """Single detection result."""
    type: DetectionType
    label: str
    confidence: float
    bbox: BoundingBox
    # Face-specific
    identity: str | None = None
    age: int | None = None
    gender: str | None = None
    emotion: str | None = None
    embedding: list[float] | None = None
    # Pose/Hand
    keypoints: list[tuple[float, float, float]] | None = None  # (x, y, confidence)
    # Screen UI
    element_type: str | None = None
    text_content: str | None = None
    # Metadata
    source: VisionSource = VisionSource.WEBCAM
    timestamp: float = field(default_factory=time.time)
    detection_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class VisionFrame:
    """Complete frame analysis result."""
    frame_id: str
    source: VisionSource
    width: int
    height: int
    detections: list[Detection]
    processing_time: float
    timestamp: float = field(default_factory=time.time)
    raw_frame: np.ndarray | None = None  # Optional, for display


@dataclass
class VisionConfig:
    """Vision engine configuration."""
    # General
    enabled: bool = True
    max_fps: int = 30
    process_every_n_frames: int = 1

    # YOLOv8 (ultralytics)
    yolo: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "model": "yolov8x.pt",
        "confidence": 0.5,
        "iou": 0.45,
        "max_det": 100,
        "classes": None,
        "device": "cuda",
        "half": True,
        "tracker": "bytetrack.yaml",
        "track_conf": 0.5,
    })

    # Face Recognition (DeepFace/InsightFace)
    face: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "model": "Facenet512",
        "detector": "retinaface",
        "threshold": 0.6,
        "align": True,
        "emotion": True,
        "age_gender": True,
        "distance_metric": "cosine",
        "db_path": str(Path.home() / ".vega" / "face_db"),
    })

    # MediaPipe
    mediapipe: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "hands": True,
        "pose": False,
        "face_mesh": False,
        "holistic": False,
        "max_num_hands": 2,
        "min_detection_confidence": 0.7,
        "min_tracking_confidence": 0.5,
        "model_complexity": 1,
    })

    # Screen Capture
    screen: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "monitor": 0,
        "fps": 10,
        "region": None,  # (x, y, w, h) or None for full
        "cursor": True,
    })

    # OCR
    ocr: dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "engine": "tesseract",  # tesseract, easyocr, paddleocr
        "languages": ["eng"],
        "confidence_threshold": 0.5,
    })

    # OmniParser (Screen UI)
    omniparser: dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,  # Requires separate service
        "endpoint": "http://vega-screen-parser:8083",
        "confidence": 0.3,
    })

    # Webcam
    webcam: dict[str, Any] = field(default_factory=lambda: {
        "device": 0,
        "width": 1280,
        "height": 720,
        "fps": 30,
        "format": "MJPG",
    })


# =============================================================================
# BASE DETECTOR INTERFACE
# =============================================================================

class BaseDetector(ABC):
    """Abstract base for vision detectors."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> bool:
        pass

    @abstractmethod
    async def detect(self, frame: np.ndarray) -> list[Detection]:
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        pass

    @property
    def is_ready(self) -> bool:
        return self._initialized


# =============================================================================
# YOLOv8 DETECTOR
# =============================================================================

class YOLODetector(BaseDetector):
    """YOLOv8 object detection, tracking, pose estimation."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._model: Any = None
        self._tracker: Any = None

    async def initialize(self) -> bool:
        try:
            from ultralytics import YOLO

            model_path = self.config.get("model", "yolov8x.pt")
            device = self.config.get("device", "cuda")

            logger.info(f"Loading YOLOv8 model: {model_path} on {device}")
            self._model = YOLO(model_path)
            self._model.to(device)

            # Warmup
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self._model(dummy, verbose=False)

            self._initialized = True
            logger.info("YOLODetector initialized")
            return True
        except ImportError:
            logger.error("ultralytics not installed. Install: pip install ultralytics")
            return False
        except Exception as e:
            logger.exception(f"YOLODetector init error: {e}")
            return False

    async def detect(self, frame: np.ndarray) -> list[Detection]:
        if not self._initialized or self._model is None:
            return []

        try:
            # Run inference
            results = self._model.track(
                frame,
                conf=self.config.get("confidence", 0.5),
                iou=self.config.get("iou", 0.45),
                max_det=self.config.get("max_det", 100),
                classes=self.config.get("classes"),
                tracker=self.config.get("tracker", "bytetrack.yaml"),
                persist=True,
                verbose=False,
                half=self.config.get("half", True),
            )

            detections = []
            for result in results:
                if result.boxes is None:
                    continue

                boxes = result.boxes
                for i in range(len(boxes)):
                    xywh = boxes.xywhn[i].cpu().numpy()  # normalized
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls = int(boxes.cls[i].cpu().numpy())
                    track_id = int(boxes.id[i].cpu().numpy()) if boxes.id is not None else None

                    label = self._model.names.get(cls, f"class_{cls}")

                    detections.append(Detection(
                        type=DetectionType.OBJECT,
                        label=f"{label}_{track_id}" if track_id else label,
                        confidence=conf,
                        bbox=BoundingBox(
                            x=float(xywh[0]),
                            y=float(xywh[1]),
                            width=float(xywh[2]),
                            height=float(xywh[3]),
                        ),
                    ))

            return detections

        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return []

    async def shutdown(self) -> None:
        self._model = None
        self._initialized = False


# =============================================================================
# FACE DETECTOR (DeepFace/InsightFace)
# =============================================================================

class FaceDetector(BaseDetector):
    """Face detection, recognition, emotion, age/gender using DeepFace or InsightFace."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._model_name = config.get("model", "Facenet512")
        self._detector_backend = config.get("detector", "retinaface")
        self._threshold = config.get("threshold", 0.6)
        self._db_path = Path(config.get("db_path", "face_db"))
        self._db_path.mkdir(parents=True, exist_ok=True)

        self._known_faces: dict[str, np.ndarray] = {}  # name -> embedding
        self._face_analyzer: Any = None

    async def initialize(self) -> bool:
        try:
            # Try InsightFace first (faster), fallback to DeepFace
            try:
                import insightface
                self._face_analyzer = insightface.app.FaceAnalysis(
                    name="buffalo_l",
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                self._face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
                self._backend = "insightface"
                logger.info("FaceDetector initialized with InsightFace")
            except ImportError:
                from deepface import DeepFace
                self._deepface = DeepFace
                self._backend = "deepface"
                logger.info("FaceDetector initialized with DeepFace")

            # Load known faces
            await self._load_known_faces()

            self._initialized = True
            return True
        except Exception as e:
            logger.exception(f"FaceDetector init error: {e}")
            return False

    async def _load_known_faces(self) -> None:
        """Load face embeddings from database."""
        self._known_faces.clear()

        for img_path in self._db_path.glob("*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                name = img_path.stem
                try:
                    embedding = await self._get_embedding(str(img_path))
                    if embedding is not None:
                        self._known_faces[name] = embedding
                        logger.debug(f"Loaded face: {name}")
                except Exception as e:
                    logger.warning(f"Failed to load face {name}: {e}")

        logger.info(f"Loaded {len(self._known_faces)} known faces")

    async def _get_embedding(self, img_path: str) -> np.ndarray | None:
        """Extract face embedding from image."""
        try:
            if self._backend == "insightface":
                img = cv2.imread(img_path)
                faces = self._face_analyzer.get(img)
                if faces:
                    return faces[0].embedding
            else:
                from deepface import DeepFace
                result = DeepFace.represent(
                    img_path=img_path,
                    model_name=self._model_name,
                    detector_backend=self._detector_backend,
                    enforce_detection=False,
                )
                if result:
                    return np.array(result[0]["embedding"])
        except Exception:
            pass
        return None

    async def detect(self, frame: np.ndarray) -> list[Detection]:
        if not self._initialized:
            return []

        detections = []

        try:
            if self._backend == "insightface":
                faces = self._face_analyzer.get(frame)
                for face in faces:
                    bbox = face.bbox.astype(float)  # [x1, y1, x2, y2]
                    h, w = frame.shape[:2]

                    # Recognition
                    identity = "unknown"
                    best_dist = float("inf")
                    for name, known_emb in self._known_faces.items():
                        dist = np.linalg.norm(face.embedding - known_emb)
                        if dist < best_dist and dist < self._threshold:
                            best_dist = dist
                            identity = name

                    # Age/Gender
                    age = int(face.age) if hasattr(face, "age") else None
                    gender = "M" if face.gender == 1 else "F" if hasattr(face, "gender") else None

                    detections.append(Detection(
                        type=DetectionType.FACE,
                        label=identity,
                        confidence=face.det_score,
                        bbox=BoundingBox(
                            x=bbox[0] / w,
                            y=bbox[1] / h,
                            width=(bbox[2] - bbox[0]) / w,
                            height=(bbox[3] - bbox[1]) / h,
                        ),
                        identity=identity,
                        age=age,
                        gender=gender,
                        embedding=face.embedding.tolist(),
                    ))

            else:
                # DeepFace detection
                from deepface import DeepFace
                results = DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend=self._detector_backend,
                    enforce_detection=False,
                    align=True,
                )

                for result in results:
                    facial_area = result["facial_area"]
                    h, w = frame.shape[:2]

                    detections.append(Detection(
                        type=DetectionType.FACE,
                        label="face",
                        confidence=result.get("confidence", 0.9),
                        bbox=BoundingBox(
                            x=facial_area["x"] / w,
                            y=facial_area["y"] / h,
                            width=facial_area["w"] / w,
                            height=facial_area["h"] / h,
                        ),
                    ))

        except Exception as e:
            logger.error(f"Face detection error: {e}")

        return detections

    async def enroll_face(self, name: str, frame: np.ndarray) -> bool:
        """Enroll a new face from current frame."""
        try:
            if self._backend == "insightface":
                faces = self._face_analyzer.get(frame)
                if faces:
                    embedding = faces[0].embedding
                    self._known_faces[name] = embedding

                    # Save image
                    save_path = self._db_path / f"{name}.jpg"
                    cv2.imwrite(str(save_path), frame)
                    logger.info(f"Enrolled face: {name}")
                    return True
            else:
                from deepface import DeepFace
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                    cv2.imwrite(f.name, frame)
                    result = DeepFace.represent(
                        img_path=f.name,
                        model_name=self._model_name,
                        detector_backend=self._detector_backend,
                    )
                    if result:
                        self._known_faces[name] = np.array(result[0]["embedding"])
                        # Save
                        save_path = self._db_path / f"{name}.jpg"
                        cv2.imwrite(str(save_path), frame)
                        logger.info(f"Enrolled face: {name}")
                        return True
        except Exception as e:
            logger.error(f"Face enrollment error: {e}")

        return False

    async def shutdown(self) -> None:
        self._face_analyzer = None
        self._initialized = False


# =============================================================================
# MEDIAPIPE DETECTOR (Hands, Pose, Face Mesh)
# =============================================================================

class MediaPipeDetector(BaseDetector):
    """MediaPipe hands, pose, face mesh, holistic tracking."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._hands: Any = None
        self._pose: Any = None
        self._face_mesh: Any = None
        self._holistic: Any = None

        self._enable_hands = config.get("hands", True)
        self._enable_pose = config.get("pose", False)
        self._enable_face_mesh = config.get("face_mesh", False)
        self._enable_holistic = config.get("holistic", False)

    async def initialize(self) -> bool:
        try:
            import mediapipe as mp

            if self._enable_hands:
                self._hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=config.get("max_num_hands", 2),
                    min_detection_confidence=config.get("min_detection_confidence", 0.7),
                    min_tracking_confidence=config.get("min_tracking_confidence", 0.5),
                    model_complexity=config.get("model_complexity", 1),
                )

            if self._enable_pose:
                self._pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5,
                    model_complexity=1,
                )

            if self._enable_face_mesh:
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5,
                )

            if self._enable_holistic:
                self._holistic = mp.solutions.holistic.Holistic(
                    static_image_mode=False,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5,
                )

            self._initialized = True
            logger.info("MediaPipeDetector initialized")
            return True
        except ImportError:
            logger.error("mediapipe not installed. Install: pip install mediapipe")
            return False
        except Exception as e:
            logger.exception(f"MediaPipe init error: {e}")
            return False

    async def detect(self, frame: np.ndarray) -> list[Detection]:
        if not self._initialized:
            return []

        detections = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        try:
            # Hands
            if self._hands:
                results = self._hands.process(rgb_frame)
                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(
                        results.multi_hand_landmarks,
                        results.multi_handedness,
                    ):
                        hand_label = handedness.classification[0].label  # Left/Right

                        keypoints = []
                        for lm in hand_landmarks.landmark:
                            keypoints.append((lm.x, lm.y, lm.z))

                        # Calculate bounding box
                        xs = [kp[0] for kp in keypoints]
                        ys = [kp[1] for kp in keypoints]
                        x1, x2 = min(xs), max(xs)
                        y1, y2 = min(ys), max(ys)

                        detections.append(Detection(
                            type=DetectionType.HAND,
                            label=f"{hand_label}_hand",
                            confidence=0.9,
                            bbox=BoundingBox(x=x1, y=y1, width=x2-x1, height=y2-y1),
                            keypoints=keypoints,
                        ))

            # Pose
            if self._pose:
                results = self._pose.process(rgb_frame)
                if results.pose_landmarks:
                    keypoints = []
                    for lm in results.pose_landmarks.landmark:
                        keypoints.append((lm.x, lm.y, lm.z))

                    xs = [kp[0] for kp in keypoints]
                    ys = [kp[1] for kp in keypoints]
                    x1, x2 = min(xs), max(xs)
                    y1, y2 = min(ys), max(ys)

                    detections.append(Detection(
                        type=DetectionType.POSE,
                        label="pose",
                        confidence=0.9,
                        bbox=BoundingBox(x=x1, y=y1, width=x2-x1, height=y2-y1),
                        keypoints=keypoints,
                    ))

            # Face Mesh
            if self._face_mesh:
                results = self._face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        keypoints = []
                        for lm in face_landmarks.landmark:
                            keypoints.append((lm.x, lm.y, lm.z))

                        xs = [kp[0] for kp in keypoints]
                        ys = [kp[1] for kp in keypoints]
                        x1, x2 = min(xs), max(xs)
                        y1, y2 = min(ys), max(ys)

                        detections.append(Detection(
                            type=DetectionType.FACE,
                            label="face_mesh",
                            confidence=0.9,
                            bbox=BoundingBox(x=x1, y=y1, width=x2-x1, height=y2-y1),
                            keypoints=keypoints,
                        ))

        except Exception as e:
            logger.error(f"MediaPipe detection error: {e}")

        return detections

    async def shutdown(self) -> None:
        for attr in ["_hands", "_pose", "_face_mesh", "_holistic"]:
            obj = getattr(self, attr, None)
            if obj:
                obj.close()
        self._initialized = False


# =============================================================================
# SCREEN CAPTURE + OCR
# =============================================================================

class ScreenCapture:
    """Screen capture using mss."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._sct: Any = None
        self._monitor = config.get("monitor", 0)
        self._region = config.get("region")  # (x, y, w, h)

    async def initialize(self) -> bool:
        try:
            import mss
            self._sct = mss.mss()
            logger.info("ScreenCapture initialized")
            return True
        except ImportError:
            logger.error("mss not installed. Install: pip install mss")
            return False
        except Exception as e:
            logger.exception(f"ScreenCapture init error: {e}")
            return False

    def capture(self) -> np.ndarray | None:
        """Capture screen frame."""
        if not self._sct:
            return None

        try:
            if self._region:
                monitor = {
                    "left": self._region[0],
                    "top": self._region[1],
                    "width": self._region[2],
                    "height": self._region[3],
                }
            else:
                monitor = self._sct.monitors[self._monitor]

            screenshot = self._sct.grab(monitor)
            frame = np.array(screenshot)
            # Convert BGRA to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
        except Exception as e:
            logger.error(f"Screen capture error: {e}")
            return None

    async def shutdown(self) -> None:
        if self._sct:
            self._sct.close()


class OCREngine:
    """OCR using Tesseract, EasyOCR, or PaddleOCR."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._engine = config.get("engine", "tesseract")
        self._languages = config.get("languages", ["eng"])
        self._threshold = config.get("confidence_threshold", 0.5)
        self._reader: Any = None

    async def initialize(self) -> bool:
        try:
            if self._engine == "tesseract":
                import pytesseract
                # Verify tesseract is available
                pytesseract.get_tesseract_version()
                self._pytesseract = pytesseract
                logger.info("OCREngine initialized with Tesseract")
            elif self._engine == "easyocr":
                import easyocr
                self._reader = easyocr.Reader(self._languages, gpu=True)
                logger.info("OCREngine initialized with EasyOCR")
            elif self._engine == "paddleocr":
                from paddleocr import PaddleOCR
                self._reader = PaddleOCR(use_angle_cls=True, lang=self._languages[0], use_gpu=True)
                logger.info("OCREngine initialized with PaddleOCR")
            else:
                logger.error(f"Unknown OCR engine: {self._engine}")
                return False

            return True
        except ImportError as e:
            logger.error(f"OCR engine not available: {e}")
            return False
        except Exception as e:
            logger.exception(f"OCREngine init error: {e}")
            return False

    def extract_text(self, frame: np.ndarray) -> list[Detection]:
        """Extract text from frame."""
        detections = []

        try:
            if self._engine == "tesseract":
                detections = self._extract_tesseract(frame)
            elif self._engine in ["easyocr", "paddleocr"] and self._reader:
                detections = self._extract_deep(frame)

        except Exception as e:
            logger.error(f"OCR extraction error: {e}")

        return detections

    def _extract_tesseract(self, frame: np.ndarray) -> list[Detection]:
        """Extract using Tesseract."""
        detections = []

        try:
            data = self._pytesseract.image_to_data(
                frame,
                lang="+".join(self._languages),
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
                        detections.append(Detection(
                            type=DetectionType.TEXT,
                            label=text,
                            confidence=conf / 100.0,
                            bbox=BoundingBox(
                                x=x / w,
                                y=y / h,
                                width=w_box / w,
                                height=h_box / h,
                            ),
                            text_content=text,
                            attributes={"block_num": data["block_num"][i], "line_num": data["line_num"][i]},
                        ))

        except Exception as e:
            logger.error(f"Tesseract extraction error: {e}")

        return detections

    def _extract_deep(self, frame: np.ndarray) -> list[Detection]:
        """Extract using EasyOCR or PaddleOCR."""
        detections = []

        try:
            results = self._reader.readtext(frame)
            h, w = frame.shape[:2]

            for result in results:
                bbox_coords, text, conf = result
                if conf > self._threshold and text.strip():
                    # bbox_coords is 4 points
                    xs = [p[0] for p in bbox_coords]
                    ys = [p[1] for p in bbox_coords]
                    x1, x2 = int(min(xs)), int(max(xs))
                    y1, y2 = int(min(ys)), int(max(ys))

                    detections.append(Detection(
                        type=DetectionType.TEXT,
                        label=text,
                        confidence=conf,
                        bbox=BoundingBox(
                            x=x1 / w,
                            y=y1 / h,
                            width=(x2 - x1) / w,
                            height=(y2 - y1) / h,
                        ),
                        text_content=text,
                    ))

        except Exception as e:
            logger.error(f"Deep OCR extraction error: {e}")

        return detections

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

    async def parse_screen(self, frame: np.ndarray) -> list[Detection]:
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

    def _parse_response(self, data: dict, frame_shape: tuple) -> list[Detection]:
        """Parse OmniParser response."""
        detections = []
        h, w = frame_shape[:2]

        for item in data.get("elements", []):
            bbox = item.get("bbox", [0, 0, 0, 0])  # x1, y1, x2, y2
            x1, y1, x2, y2 = bbox

            detections.append(Detection(
                element_id=str(uuid.uuid4()),
                type=DetectionType.SCREEN_UI,
                label=item.get("type", "unknown"),
                confidence=item.get("confidence", 0.0),
                bbox=BoundingBox(
                    x=x1 / w,
                    y=y1 / h,
                    width=(x2 - x1) / w,
                    height=(y2 - y1) / h,
                ),
                text_content=item.get("text", ""),
                element_type=item.get("type", "unknown"),
                attributes=item.get("attributes", {}),
            ))

        return detections

    async def shutdown(self) -> None:
        if self._session:
            await self._session.close()


# =============================================================================
# VISION PIPELINE
# =============================================================================

class VisionPipeline:
    """
    Orchestrates multiple detectors in parallel.
    Manages frame processing, result aggregation, and callbacks.
    """

    def __init__(self, config: VisionConfig):
        self.config = config

        # Detectors
        self.yolo_detector = YOLODetector(config.yolo) if config.yolo.get("enabled") else None
        self.face_detector = FaceDetector(config.face) if config.face.get("enabled") else None
        self.mediapipe_detector = MediaPipeDetector(config.mediapipe) if config.mediapipe.get("enabled") else None
        self.screen_capture = ScreenCapture(config.screen) if config.screen.get("enabled") else None
        self.ocr_engine = OCREngine(config.ocr) if config.ocr.get("enabled") else None
        self.omniparser = OmniParserClient(
            config.omniparser.get("endpoint", "http://vega-screen-parser:8083"),
            config.omniparser.get("confidence", 0.3),
        ) if config.omniparser.get("enabled") else None

        # State
        self._initialized = False
        self._running = False
        self._process_task: asyncio.Task | None = None
        self._frame_queue: asyncio.Queue = asyncio.Queue(maxsize=5)

        # Callbacks
        self._on_frame: Callable[[VisionFrame], Any] | None = None
        self._on_detection: Callable[[Detection], Any] | None = None

        # Stats
        self._stats = {
            "frames_processed": 0,
            "total_detections": 0,
            "avg_fps": 0.0,
            "avg_processing_ms": 0.0,
        }

    async def initialize(self) -> bool:
        """Initialize all detectors."""
        tasks = []

        if self.yolo_detector:
            tasks.append(self.yolo_detector.initialize())
        if self.face_detector:
            tasks.append(self.face_detector.initialize())
        if self.mediapipe_detector:
            tasks.append(self.mediapipe_detector.initialize())
        if self.screen_capture:
            tasks.append(self.screen_capture.initialize())
        if self.ocr_engine:
            tasks.append(self.ocr_engine.initialize())
        if self.omniparser:
            tasks.append(self.omniparser.initialize())

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check results
        for i, result in enumerate(results):
            if isinstance(result, Exception) or result is False:
                logger.error(f"Detector {i} failed to initialize")

        self._initialized = True
        logger.info("VisionPipeline initialized")
        return True

    async def start(self) -> None:
        """Start processing loop."""
        if not self._initialized:
            await self.initialize()

        self._running = True
        self._process_task = asyncio.create_task(self._process_loop())
        logger.info("VisionPipeline started")

    async def stop(self) -> None:
        """Stop processing loop."""
        self._running = False

        if self._process_task:
            self._process_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._process_task

        # Shutdown detectors
        for detector in [self.yolo_detector, self.face_detector, self.mediapipe_detector]:
            if detector:
                await detector.shutdown()

        if self.screen_capture:
            await self.screen_capture.shutdown()
        if self.ocr_engine:
            await self.ocr_engine.shutdown()
        if self.omniparser:
            await self.omniparser.shutdown()

        logger.info("VisionPipeline stopped")

    async def _process_loop(self) -> None:
        """Main processing loop."""
        frame_interval = 1.0 / self.config.max_fps

        while self._running:
            start_time = time.time()

            try:
                # Capture frames from all sources
                frames = {}

                # Webcam
                webcam_frame = await self._capture_webcam()
                if webcam_frame is not None:
                    frames[VisionSource.WEBCAM] = webcam_frame

                # Screen
                if self.screen_capture:
                    screen_frame = self.screen_capture.capture()
                    if screen_frame is not None:
                        frames[VisionSource.SCREEN] = screen_frame

                # Process each frame
                for source, frame in frames.items():
                    self._frame_count += 1

                    # Skip frames if configured
                    if self._frame_count % self.config.process_every_n_frames != 0:
                        continue

                    # Run detectors in parallel
                    detections = await self._run_detectors(frame, source)

                    # Create frame result
                    vision_frame = VisionFrame(
                        frame_id=str(uuid.uuid4()),
                        source=source,
                        width=frame.shape[1],
                        height=frame.shape[0],
                        detections=detections,
                        processing_time=time.time() - start_time,
                    )

                    # Update stats
                    self._stats["frames_processed"] += 1
                    self._stats["total_detections"] += len(detections)
                    self._stats["avg_processing_ms"] = (
                        (self._stats["avg_processing_ms"] * (self._stats["frames_processed"] - 1)
                         + vision_frame.processing_time * 1000)
                        / self._stats["frames_processed"]
                    )

                    # Callbacks
                    if self._on_frame:
                        await self._on_frame(vision_frame)

                    for det in detections:
                        if self._on_detection:
                            await self._on_detection(det)

                # FPS control
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_interval - elapsed)
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Vision process loop error: {e}")
                await asyncio.sleep(0.1)

    async def _capture_webcam(self) -> np.ndarray | None:
        """Capture from webcam."""
        # This would use OpenCV VideoCapture
        # For now, return None - webcam capture runs in separate thread
        return None

    async def _run_detectors(self, frame: np.ndarray, source: VisionSource) -> list[Detection]:
        """Run all applicable detectors on frame."""
        all_detections = []

        # Determine which detectors to run based on source
        tasks = []

        if source == VisionSource.WEBCAM:
            if self.yolo_detector:
                tasks.append(self.yolo_detector.detect(frame))
            if self.face_detector:
                tasks.append(self.face_detector.detect(frame))
            if self.mediapipe_detector:
                tasks.append(self.mediapipe_detector.detect(frame))

        elif source == VisionSource.SCREEN:
            if self.ocr_engine:
                tasks.append(asyncio.to_thread(self.ocr_engine.extract_text, frame))
            if self.omniparser:
                tasks.append(self.omniparser.parse_screen(frame))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                for det in result:
                    det.source = source
                    all_detections.append(det)
            elif isinstance(result, Exception):
                logger.error(f"Detector error: {result}")

        return all_detections

    def on_frame(self, callback: Callable[[VisionFrame], Any]) -> None:
        self._on_frame = callback

    def on_detection(self, callback: Callable[[Detection], Any]) -> None:
        self._on_detection = callback

    def get_stats(self) -> dict[str, Any]:
        return self._stats.copy()

    async def enroll_face(self, name: str, frame: np.ndarray) -> bool:
        """Enroll face using face detector."""
        if self.face_detector:
            return await self.face_detector.enroll_face(name, frame)
        return False


# =============================================================================
# VISION ENGINE (High-level API)
# =============================================================================

class VisionEngine:
    """
    High-level vision engine managing pipeline, webcam, and WebSocket streaming.
    """

    def __init__(
        self,
        websocket_manager: Any = None,
        event_bus: Any = None,
        config: VisionConfig | None = None,
    ):
        self.websocket_manager = websocket_manager
        self.event_bus = event_bus
        self.config = config or VisionConfig()

        self.pipeline = VisionPipeline(self.config)
        self._webcam_task: asyncio.Task | None = None
        self._webcam_cap: Any = None
        self._running = False

    async def start(self) -> bool:
        """Start vision engine."""
        if self._running:
            return True

        # Initialize pipeline
        await self.pipeline.initialize()

        # Start webcam capture
        await self._start_webcam()

        # Start pipeline
        await self.pipeline.start()

        # Register callbacks
        self.pipeline.on_frame(self._on_frame)
        self.pipeline.on_detection(self._on_detection)

        self._running = True
        logger.info("VisionEngine started")
        return True

    async def stop(self) -> None:
        """Stop vision engine."""
        self._running = False

        if self._webcam_task:
            self._webcam_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._webcam_task

        if self._webcam_cap:
            self._webcam_cap.release()

        await self.pipeline.stop()
        logger.info("VisionEngine stopped")

    async def _start_webcam(self) -> None:
        """Start webcam capture thread."""
        wc_config = self.config.webcam

        def capture_loop():
            self._webcam_cap = cv2.VideoCapture(wc_config.get("device", 0))
            self._webcam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, wc_config.get("width", 1280))
            self._webcam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, wc_config.get("height", 720))
            self._webcam_cap.set(cv2.CAP_PROP_FPS, wc_config.get("fps", 30))

            if wc_config.get("format") == "MJPG":
                self._webcam_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

            while self._running:
                ret, frame = self._webcam_cap.read()
                if ret:
                    try:
                        self.pipeline._frame_queue.put_nowait((VisionSource.WEBCAM, frame))
                    except asyncio.QueueFull:
                        pass
                else:
                    time.sleep(0.01)

        self._webcam_task = asyncio.create_task(asyncio.to_thread(capture_loop))

    async def _on_frame(self, frame: VisionFrame) -> None:
        """Handle processed frame."""
        # Broadcast to HUD
        if self.websocket_manager:
            # Send summary (not raw frame)
            summary = {
                "frame_id": frame.frame_id,
                "source": frame.source.value,
                "detections": [
                    {
                        "type": d.type.value,
                        "label": d.label,
                        "confidence": d.confidence,
                        "bbox": {"x": d.bbox.x, "y": d.bbox.y, "w": d.bbox.width, "h": d.bbox.height},
                    }
                    for d in frame.detections[:20]  # Limit
                ],
                "processing_ms": frame.processing_time * 1000,
            }

            await self.websocket_manager.broadcast({
                "type": "vision_frame",
                "payload": summary,
            }, room="hud")

        # Emit event
        if self.event_bus:
            await self.event_bus.emit("vision.frame", {
                "source": frame.source.value,
                "detection_count": len(frame.detections),
                "processing_ms": frame.processing_time * 1000,
            })

    async def _on_detection(self, detection: Detection) -> None:
        """Handle individual detection."""
        # Broadcast specific detections
        if self.websocket_manager:
            await self.websocket_manager.broadcast({
                "type": "vision_detection",
                "payload": {
                    "type": detection.type.value,
                    "label": detection.label,
                    "confidence": detection.confidence,
                    "bbox": {"x": detection.bbox.x, "y": detection.bbox.y, "w": detection.bbox.width, "h": detection.bbox.height},
                    "metadata": {
                        "identity": detection.identity,
                        "emotion": detection.emotion,
                        "age": detection.age,
                        "gender": detection.gender,
                    },
                },
            }, room="hud")

    def get_stats(self) -> dict[str, Any]:
        return self.pipeline.get_stats()

    async def enroll_face(self, name: str) -> bool:
        """Enroll face from current webcam frame."""
        # Get latest frame from queue
        try:
            source, frame = self.pipeline._frame_queue.get_nowait()
            if source == VisionSource.WEBCAM:
                return await self.pipeline.enroll_face(name, frame)
        except asyncio.QueueEmpty:
            pass
        return False


# =============================================================================
# FACTORY
# =============================================================================

async def create_vision_engine(
    websocket_manager: Any = None,
    event_bus: Any = None,
    config: VisionConfig | None = None,
) -> VisionEngine:
    """Factory to create and start vision engine."""
    engine = VisionEngine(
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
    "VisionEngine",
    "VisionPipeline",
    "VisionConfig",
    "VisionSource",
    "DetectionType",
    "BoundingBox",
    "Detection",
    "VisionFrame",
    "YOLODetector",
    "FaceDetector",
    "MediaPipeDetector",
    "ScreenCapture",
    "OCREngine",
    "OmniParserClient",
    "create_vision_engine",
]