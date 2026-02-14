"""Visualization utilities for easy_rfdetr."""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional


NEON_PALETTE = [
    (255, 0, 255),
    (0, 255, 255),
    (255, 165, 0),
    (50, 205, 50),
    (255, 20, 147),
    (0, 191, 255),
    (255, 215, 0),
    (138, 43, 226),
    (0, 255, 127),
    (220, 20, 60),
    (64, 224, 208),
    (255, 105, 180),
]


def get_color_palette(n: int) -> List[Tuple[int, int, int]]:
    """Generate color palette for n classes.

    Args:
        n: Number of colors needed

    Returns:
        List of RGB color tuples
    """
    if n <= len(NEON_PALETTE):
        return NEON_PALETTE[:n]

    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        colors.append(hsv_to_rgb(hue, 1.0, 1.0))
    return colors


def hsv_to_rgb(h: int, s: float, v: float) -> Tuple[int, int, int]:
    """Convert HSV to RGB color.

    Args:
        h: Hue (0-180)
        s: Saturation (0-1)
        v: Value (0-1)

    Returns:
        RGB tuple
    """
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(h / 180, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def draw_boxes(
    image: np.ndarray,
    boxes: torch.Tensor,
    labels: List[str],
    scores: torch.Tensor,
    color_palette: Optional[List[Tuple[int, int, int]]] = None,
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on image.

    Args:
        image: Input image (H, W, 3)
        boxes: Bounding boxes in xyxy format (N, 4)
        labels: Class labels (N,)
        scores: Confidence scores (N,)
        color_palette: Custom color palette
        thickness: Box line thickness

    Returns:
        Annotated image
    """
    if len(boxes) == 0:
        return image

    img = image.copy()

    if color_palette is None:
        num_classes = len(set(labels))
        color_palette = get_color_palette(max(num_classes, 1))

    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        x1, y1, x2, y2 = box.int().tolist()
        color = color_palette[i % len(color_palette)]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        label_text = f"{label}: {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        text_x = x1
        text_y = y1 - 10
        if text_y - text_height < 0:
            text_y = y1 + text_height + 10

        cv2.rectangle(
            img,
            (text_x, text_y - text_height - baseline),
            (text_x + text_width, text_y),
            color,
            -1,
        )

        cv2.putText(
            img,
            label_text,
            (text_x, text_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return img


def annotate_image(
    image: np.ndarray,
    boxes: torch.Tensor,
    labels: List[str],
    scores: torch.Tensor,
    color_palette: Optional[List[Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """Annotate image with detection results (alias for draw_boxes).

    Args:
        image: Input image (H, W, 3)
        boxes: Bounding boxes in xyxy format (N, 4)
        labels: Class labels (N,)
        scores: Confidence scores (N,)
        color_palette: Custom color palette

    Returns:
        Annotated image
    """
    return draw_boxes(image, boxes, labels, scores, color_palette)


def create_strip(images: List[np.ndarray], horizontal: bool = True) -> np.ndarray:
    """Create image strip from list of images.

    Args:
        images: List of images
        horizontal: If True, stack horizontally, else vertically

    Returns:
        Combined image strip
    """
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    if horizontal:
        max_height = max(img.shape[0] for img in images)
        resized = []
        for img in images:
            if img.shape[0] != max_height:
                h, w = img.shape[:2]
                ratio = max_height / h
                new_w = int(w * ratio)
                img = cv2.resize(img, (new_w, max_height))
            resized.append(img)
        return np.hstack(resized)
    else:
        max_width = max(img.shape[1] for img in images)
        resized = []
        for img in images:
            if img.shape[1] != max_width:
                h, w = img.shape[:2]
                ratio = max_width / w
                new_h = int(h * ratio)
                img = cv2.resize(img, (max_width, new_h))
            resized.append(img)
        return np.vstack(resized)
