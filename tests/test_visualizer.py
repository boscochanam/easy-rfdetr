"""Tests for easy_rfdetr visualizer module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np


class TestVisualizer:
    """Tests for visualizer functionality."""

    def test_draw_boxes_function_exists(self):
        """Test that draw_boxes function can be imported."""
        from easy_rfdetr.visualizer import draw_boxes

        assert draw_boxes is not None

    def test_get_color_palette(self):
        """Test color palette generation."""
        from easy_rfdetr.visualizer import get_color_palette

        colors = get_color_palette(10)
        assert len(colors) == 10

    def test_draw_boxes_with_empty_boxes(self):
        """Test drawing boxes when there are no boxes."""
        from easy_rfdetr.visualizer import draw_boxes

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = torch.tensor([])
        labels = []
        scores = torch.tensor([])

        result = draw_boxes(img, boxes, labels, scores)
        assert result is not None


class TestAnnotateImage:
    """Tests for image annotation."""

    def test_annotate_function_exists(self):
        """Test annotate function exists."""
        from easy_rfdetr.visualizer import annotate_image

        assert annotate_image is not None

    def test_annotate_with_empty_boxes(self):
        """Test annotating image with no detections."""
        from easy_rfdetr.visualizer import annotate_image

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes = torch.tensor([])
        labels = []
        scores = torch.tensor([])

        result = annotate_image(img, boxes, labels, scores)
        assert result is not None


class TestNeonPalette:
    """Tests for neon color palette."""

    def test_neon_palette_exists(self):
        """Test neon palette is defined."""
        from easy_rfdetr.visualizer import NEON_PALETTE

        assert len(NEON_PALETTE) > 0

    def test_palette_colors_are_rgb(self):
        """Test palette colors are in RGB format."""
        from easy_rfdetr.visualizer import NEON_PALETTE

        for color in NEON_PALETTE:
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)
