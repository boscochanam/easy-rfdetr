"""Results class for easy_rfdetr."""

import os
from typing import List, Optional, Union
from pathlib import Path

import torch
import numpy as np
from PIL import Image


class Results:
    """Container for detection results.

    Attributes:
        boxes: Bounding boxes in xyxy format (N, 4)
        scores: Confidence scores (N,)
        class_ids: Class indices (N,)
        class_names: Human-readable class names (N,)
    """

    def __init__(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        class_ids: torch.Tensor,
        class_names: Optional[List[str]] = None,
        original_image: Optional[Image.Image] = None,
    ):
        """Initialize Results with detection data.

        Args:
            boxes: Bounding boxes in xyxy format
            scores: Confidence scores
            class_ids: Class indices
            class_names: Optional human-readable class names
            original_image: Original PIL Image for visualization
        """
        self._boxes = boxes
        self._scores = scores
        self._class_ids = class_ids
        self._class_names = class_names or [str(cid.item()) for cid in class_ids]
        self._original_image = original_image

    @property
    def boxes(self) -> torch.Tensor:
        """Get bounding boxes in xyxy format."""
        return self._boxes

    @property
    def scores(self) -> torch.Tensor:
        """Get confidence scores."""
        return self._scores

    @property
    def class_ids(self) -> torch.Tensor:
        """Get class indices."""
        return self._class_ids

    @property
    def labels(self) -> List[str]:
        """Get human-readable class labels."""
        return self._class_names

    @property
    def original_image(self) -> Optional[Image.Image]:
        """Get original image."""
        return self._original_image

    def __len__(self) -> int:
        """Get number of detections."""
        return len(self._boxes)

    def __getitem__(self, idx: int) -> dict:
        """Get single detection by index."""
        return {
            "box": self._boxes[idx].tolist(),
            "score": self._scores[idx].item(),
            "class_id": self._class_ids[idx].item(),
            "label": self._class_names[idx],
        }

    def filter(self, threshold: float = 0.5) -> "Results":
        """Filter detections by confidence threshold.

        Args:
            threshold: Minimum confidence score

        Returns:
            Filtered Results object
        """
        if len(self._boxes) == 0:
            return self

        mask = self._scores >= threshold
        return Results(
            boxes=self._boxes[mask],
            scores=self._scores[mask],
            class_ids=self._class_ids[mask],
            class_names=[n for n, m in zip(self._class_names, mask) if m],
            original_image=self._original_image,
        )

    def save(self, path: Union[str, Path], annotator=None) -> str:
        """Save annotated image to disk.

        Args:
            path: Output file path
            annotator: Optional custom annotator function

        Returns:
            Path to saved file
        """
        from easy_rfdetr.visualizer import annotate_image

        if self._original_image is None:
            raise ValueError("No original image available for saving")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        img_array = np.array(self._original_image)

        if len(self._boxes) > 0:
            annotated = annotate_image(
                img_array,
                self._boxes,
                self._class_names,
                self._scores,
            )
        else:
            annotated = img_array

        import cv2

        cv2.imwrite(str(path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        print(f"✅ Saved results to {path}")
        return str(path)

    def show(self, annotator=None) -> None:
        """Display annotated image.

        Args:
            annotator: Optional custom annotator function
        """
        if self._original_image is None:
            print("⚠️  No image to display")
            return

        from easy_rfdetr.visualizer import annotate_image

        img_array = np.array(self._original_image)

        if len(self._boxes) > 0:
            annotated = annotate_image(
                img_array,
                self._boxes,
                self._class_names,
                self._scores,
            )
        else:
            annotated = img_array

        try:
            import cv2

            cv2.imshow("RFDETR Detection", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"⚠️  Could not display image: {e}")
            annotated_pil = Image.fromarray(annotated)
            annotated_pil.show()

    def to_coco(self) -> dict:
        """Convert results to COCO format.

        Returns:
            Dictionary with boxes, scores, and categories
        """
        return {
            "boxes": self._boxes.cpu().numpy().tolist(),
            "scores": self._scores.cpu().numpy().tolist(),
            "class_ids": self._class_ids.cpu().numpy().tolist(),
            "labels": self._class_names,
        }
