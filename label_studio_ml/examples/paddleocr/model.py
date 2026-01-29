"""
PaddleOCR v5 ML Backend for Label Studio

This module implements a Label Studio ML backend that uses PP-OCRv5 for
text detection and recognition, providing pre-annotations for OCR tasks.

Environment Variables:
    IMAGE_DATA_KEY: Key in task["data"] containing the image reference (default: "image")
    IMAGE_TO_NAME: <Image name="..."> in labeling config (default: "image")
    BBOX_FROM_NAME: <Rectangle name="..."> in labeling config (default: "bbox")
    LABEL_FROM_NAME: <Labels name="..."> in labeling config (default: "label")
    TEXT_FROM_NAME: <TextArea name="..."> in labeling config (default: "transcription")
    PREDICTED_LABEL: Label value from <Label value="..."> (default: "Text")
    PADDLEOCR_VERSION: OCR version - "PP-OCRv5" or "PP-OCRv4" (default: "PP-OCRv5")
    PADDLEOCR_MODEL_TYPE: Model type - "server" or "mobile" (default: "server")
    PADDLEOCR_LANG: Language code (default: "ch")
    PADDLEOCR_USE_GPU: Whether to use GPU (default: "false")
    PADDLEOCR_USE_DOC_ORIENTATION: Enable document orientation classification (default: "false")
    PADDLEOCR_USE_DOC_UNWARPING: Enable document unwarping (default: "false")
    PADDLEOCR_USE_TEXTLINE_ORIENTATION: Enable text line orientation (default: "true")
    PADDLEOCR_SCORE_THRESHOLD: Minimum confidence score threshold (default: "0.5")
"""

import os
import uuid
import logging
from typing import Any, Dict, List, Optional, Tuple

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_ml.utils import get_image_size
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path

logger = logging.getLogger(__name__)


def quad_to_rect_percent(
    quad: List[List[float]], image_width: int, image_height: int
) -> Tuple[float, float, float, float]:
    """
    Convert PaddleOCR quadrilateral (4 points) to Label Studio rectangle
    in percentage coordinates: x, y, width, height (0..100).

    Args:
        quad: List of 4 [x, y] points representing the quadrilateral
        image_width: Width of the original image in pixels
        image_height: Height of the original image in pixels

    Returns:
        Tuple of (x, y, width, height) in percentage (0-100)
    """
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]

    x_min = max(min(xs), 0)
    y_min = max(min(ys), 0)
    x_max = min(max(xs), image_width)
    y_max = min(max(ys), image_height)

    # Guard against degenerate boxes
    if x_max <= x_min or y_max <= y_min:
        return 0.0, 0.0, 0.0, 0.0

    x = x_min / image_width * 100.0
    y = y_min / image_height * 100.0
    w = (x_max - x_min) / image_width * 100.0
    h = (y_max - y_min) / image_height * 100.0

    return x, y, w, h


def get_env_bool(key: str, default: str = "false") -> bool:
    """Get boolean value from environment variable."""
    return os.getenv(key, default).lower() in ("true", "1", "yes")


class PaddleOCR(LabelStudioMLBase):
    """
    PaddleOCR v5 ML backend for Label Studio.

    This backend uses PP-OCRv5 (or PP-OCRv4) for text detection and recognition,
    returning bounding boxes and transcriptions as Label Studio predictions.

    Expected labeling config pattern (names must match environment variables):
        - <Image name="image" value="$image"/>
        - <Rectangle name="bbox" toName="image"/>
        - <Labels name="label" toName="image"> ... </Labels>
        - <TextArea name="transcription" toName="image" perRegion="true"/>
    """

    # Label Studio field configuration (from environment variables)
    IMAGE_DATA_KEY = os.getenv("IMAGE_DATA_KEY", "image")
    IMAGE_TO_NAME = os.getenv("IMAGE_TO_NAME", "image")
    BBOX_FROM_NAME = os.getenv("BBOX_FROM_NAME", "bbox")
    LABEL_FROM_NAME = os.getenv("LABEL_FROM_NAME", "label")
    TEXT_FROM_NAME = os.getenv("TEXT_FROM_NAME", "transcription")
    PREDICTED_LABEL = os.getenv("PREDICTED_LABEL", "Text")
    SCORE_THRESHOLD = float(os.getenv("PADDLEOCR_SCORE_THRESHOLD", "0.5"))

    # PaddleOCR configuration (from environment variables)
    OCR_VERSION = os.getenv("PADDLEOCR_VERSION", "PP-OCRv5")
    MODEL_TYPE = os.getenv("PADDLEOCR_MODEL_TYPE", "server")
    LANG = os.getenv("PADDLEOCR_LANG", "ch")
    USE_GPU = get_env_bool("PADDLEOCR_USE_GPU", "false")
    USE_DOC_ORIENTATION = get_env_bool("PADDLEOCR_USE_DOC_ORIENTATION", "false")
    USE_DOC_UNWARPING = get_env_bool("PADDLEOCR_USE_DOC_UNWARPING", "false")
    USE_TEXTLINE_ORIENTATION = get_env_bool("PADDLEOCR_USE_TEXTLINE_ORIENTATION", "true")

    # Label Studio connection settings
    LABEL_STUDIO_ACCESS_TOKEN = (
        os.environ.get("LABEL_STUDIO_ACCESS_TOKEN") or os.environ.get("LABEL_STUDIO_API_KEY")
    )
    LABEL_STUDIO_HOST = (
        os.environ.get("LABEL_STUDIO_HOST") or os.environ.get("LABEL_STUDIO_URL")
    )

    MODEL_DIR = os.getenv("MODEL_DIR", ".")

    # Lazy-loaded OCR model
    _ocr = None

    def _lazy_init(self):
        """Initialize PaddleOCR model on first use (lazy loading)."""
        if self._ocr is not None:
            return

        try:
            from paddleocr import PaddleOCR as PaddleOCREngine

            self._ocr = PaddleOCREngine(
                ocr_version=self.OCR_VERSION,
                lang=self.LANG,
                use_doc_orientation_classify=self.USE_DOC_ORIENTATION,
                use_doc_unwarping=self.USE_DOC_UNWARPING,
                use_textline_orientation=self.USE_TEXTLINE_ORIENTATION,
                device="gpu" if self.USE_GPU else "cpu",
            )
            logger.info(
                f"Initialized PaddleOCR with version={self.OCR_VERSION}, "
                f"model_type={self.MODEL_TYPE}, lang={self.LANG}, gpu={self.USE_GPU}"
            )
        except ImportError:
            logger.error(
                "Failed to import paddleocr. Please install it with: "
                "pip install paddleocr paddlepaddle"
            )
            raise

    def setup(self):
        """Configure model parameters."""
        model_version = (
            f"paddleocr-{self.OCR_VERSION.lower().replace('-', '')}-{self.MODEL_TYPE}"
            f"(lang={self.LANG},gpu={self.USE_GPU})"
        )
        self.set("model_version", model_version)

    def _get_model_names(self) -> Tuple[str, str]:
        """
        Get detection and recognition model names based on configuration.

        Returns:
            Tuple of (detection_model_name, recognition_model_name)
        """
        version = self.OCR_VERSION.replace("-", "").lower()  # PP-OCRv5 -> ppocrv5

        if self.MODEL_TYPE == "server":
            det_model = f"{version}_server_det"
            rec_model = f"{version}_server_rec"
        else:
            det_model = f"{version}_mobile_det"
            rec_model = f"{version}_mobile_rec"

        return det_model, rec_model

    def predict(
        self, tasks: List[Dict[str, Any]], context: Optional[Dict] = None, **kwargs
    ) -> ModelResponse:
        """
        Run OCR prediction on a list of tasks.

        Args:
            tasks: List of Label Studio tasks, each containing image data
            context: Optional context information
            **kwargs: Additional keyword arguments

        Returns:
            ModelResponse with OCR predictions
        """
        self._lazy_init()
        predictions = []

        for task in tasks:
            try:
                result = self._predict_single_task(task)
                if result:
                    predictions.append(result)
            except Exception as e:
                logger.error(f"Error processing task {task.get('id')}: {e}")
                predictions.append({
                    "result": [],
                    "model_version": self.get("model_version"),
                    "error": str(e),
                })

        return ModelResponse(predictions=predictions, model_version=self.get("model_version"))

    def _predict_single_task(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single task and return OCR predictions.

        Args:
            task: Label Studio task containing image data

        Returns:
            Prediction dictionary with OCR results
        """
        data = task.get("data", {})
        image_ref = data.get(self.IMAGE_DATA_KEY)

        if not image_ref:
            logger.warning(f"No image found in task {task.get('id')}")
            return {"result": [], "model_version": self.get("model_version")}

        # Handle local file paths directly
        if os.path.isabs(image_ref) and os.path.exists(image_ref):
            image_path = image_ref
        else:
            # Resolve Label Studio task reference to local filepath
            cache_dir = os.path.join(self.MODEL_DIR, ".file-cache")
            os.makedirs(cache_dir, exist_ok=True)

            image_path = get_local_path(
                image_ref,
                cache_dir=cache_dir,
                hostname=self.LABEL_STUDIO_HOST,
                access_token=self.LABEL_STUDIO_ACCESS_TOKEN,
                task_id=task.get("id"),
            )

        # Get image dimensions
        img_width, img_height = get_image_size(image_path)

        # Run PaddleOCR
        ocr_result = self._ocr.predict(image_path)

        # Process OCR output
        results: List[Dict[str, Any]] = []
        all_scores = []

        # Handle the result structure from PaddleOCR 3.x
        # OCRResult is a dictionary-like object with keys: rec_texts, rec_scores, dt_polys, etc.
        if ocr_result:
            for res in ocr_result:
                # Access the OCR result data using dictionary-style access
                texts = res.get("rec_texts", []) if hasattr(res, "get") else getattr(res, "rec_texts", [])
                polys = res.get("dt_polys", []) if hasattr(res, "get") else getattr(res, "dt_polys", [])
                scores = res.get("rec_scores", []) if hasattr(res, "get") else getattr(res, "rec_scores", [])

                if not texts:
                    logger.debug(f"No text found in result: {type(res)}")
                    continue

                for text, poly, score in zip(texts, polys, scores):
                    if score < self.SCORE_THRESHOLD:
                        continue

                    # Convert polygon to rectangle
                    quad = poly.tolist() if hasattr(poly, "tolist") else poly
                    x, y, w, h = quad_to_rect_percent(quad, img_width, img_height)

                    if w <= 0 or h <= 0:
                        continue

                    region_id = str(uuid.uuid4())[:10]

                    # Add rectangle, label, and transcription results
                    results.extend(
                        self._create_region_results(
                            region_id, x, y, w, h, img_width, img_height, text, float(score)
                        )
                    )
                    all_scores.append(float(score))

        avg_score = sum(all_scores) / max(len(all_scores), 1) if all_scores else 0.0

        return {
            "result": results,
            "score": avg_score,
            "model_version": self.get("model_version"),
        }

    def _create_region_results(
        self,
        region_id: str,
        x: float,
        y: float,
        w: float,
        h: float,
        original_width: int,
        original_height: int,
        text: str,
        score: float,
    ) -> List[Dict[str, Any]]:
        """
        Create Label Studio result entries for a single OCR region.

        Args:
            region_id: Unique identifier for this region
            x, y, w, h: Rectangle coordinates in percentage (0-100)
            original_width, original_height: Original image dimensions
            text: Recognized text
            score: Confidence score

        Returns:
            List of result dictionaries (rectangle, labels, textarea)
        """
        common_value = {
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "rotation": 0,
        }

        results = [
            # Rectangle geometry
            {
                "id": region_id,
                "from_name": self.BBOX_FROM_NAME,
                "to_name": self.IMAGE_TO_NAME,
                "type": "rectangle",
                "original_width": original_width,
                "original_height": original_height,
                "image_rotation": 0,
                "value": common_value,
                "score": score,
            },
            # Label attached to the same region
            {
                "id": region_id,
                "from_name": self.LABEL_FROM_NAME,
                "to_name": self.IMAGE_TO_NAME,
                "type": "labels",
                "original_width": original_width,
                "original_height": original_height,
                "image_rotation": 0,
                "value": {
                    **common_value,
                    "labels": [self.PREDICTED_LABEL],
                },
                "score": score,
            },
            # Transcription attached to the same region
            {
                "id": region_id,
                "from_name": self.TEXT_FROM_NAME,
                "to_name": self.IMAGE_TO_NAME,
                "type": "textarea",
                "original_width": original_width,
                "original_height": original_height,
                "image_rotation": 0,
                "value": {
                    **common_value,
                    "text": [text],
                },
                "score": score,
            },
        ]

        return results

    def fit(self, event: str, data: Dict, **kwargs) -> Dict[str, str]:
        """
        Handle training/fit events from Label Studio.

        This is an inference-only backend, so we just acknowledge the event.

        Args:
            event: Event type (e.g., "ANNOTATION_CREATED")
            data: Event data
            **kwargs: Additional keyword arguments

        Returns:
            Status dictionary
        """
        logger.info(f"Received fit event: {event}")
        return {"status": "ok"}
