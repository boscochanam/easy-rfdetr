"""Tests for easy_rfdetr Results class."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np
from PIL import Image


class TestResultsInit:
    """Tests for Results class initialization."""

    def test_results_class_exists(self):
        """Test that Results class can be imported."""
        from easy_rfdetr.results import Results

        assert Results is not None

    def test_results_init_with_data(self):
        """Test Results initialization with detection data."""
        from easy_rfdetr.results import Results

        results = Results(
            boxes=torch.tensor([[10, 10, 50, 50], [100, 100, 200, 200]]),
            scores=torch.tensor([0.9, 0.8]),
            class_ids=torch.tensor([0, 1]),
            class_names=["person", "car"],
        )

        assert results is not None
        assert len(results.boxes) == 2

    def test_results_empty_init(self):
        """Test Results initialization with empty data."""
        from easy_rfdetr.results import Results

        results = Results(
            boxes=torch.tensor([]),
            scores=torch.tensor([]),
            class_ids=torch.tensor([]),
            class_names=[],
        )

        assert len(results.boxes) == 0


class TestResultsProperties:
    """Tests for Results class properties."""

    def test_boxes_property(self):
        """Test boxes property returns correct data."""
        from easy_rfdetr.results import Results

        boxes = torch.tensor([[10, 10, 50, 50], [100, 100, 200, 200]])
        results = Results(
            boxes=boxes,
            scores=torch.tensor([0.9, 0.8]),
            class_ids=torch.tensor([0, 1]),
            class_names=["person", "car"],
        )

        assert torch.equal(results.boxes, boxes)

    def test_scores_property(self):
        """Test scores property returns correct data."""
        from easy_rfdetr.results import Results

        scores = torch.tensor([0.9, 0.8])
        results = Results(
            boxes=torch.tensor([[10, 10, 50, 50]]),
            scores=scores,
            class_ids=torch.tensor([0]),
            class_names=["person"],
        )

        assert torch.equal(results.scores, scores)

    def test_labels_property(self):
        """Test labels property returns human-readable class names."""
        from easy_rfdetr.results import Results

        results = Results(
            boxes=torch.tensor([[10, 10, 50, 50]]),
            scores=torch.tensor([0.9]),
            class_ids=torch.tensor([0]),
            class_names=["person"],
        )

        assert results.labels == ["person"]

    def test_labels_from_class_ids(self):
        """Test labels property falls back to class IDs when names not provided."""
        from easy_rfdetr.results import Results

        results = Results(
            boxes=torch.tensor([[10, 10, 50, 50]]),
            scores=torch.tensor([0.9]),
            class_ids=torch.tensor([0]),
            class_names=None,
        )

        assert results.labels == ["0"]

    def test_xyxy_format(self):
        """Test boxes are in xyxy format."""
        from easy_rfdetr.results import Results

        results = Results(
            boxes=torch.tensor([[10, 20, 50, 60]]),
            scores=torch.tensor([0.9]),
            class_ids=torch.tensor([0]),
            class_names=["person"],
        )

        box = results.boxes[0]
        assert box[0] == 10  # x1
        assert box[1] == 20  # y1
        assert box[2] == 50  # x2
        assert box[3] == 60  # y2


class TestResultsFiltering:
    """Tests for Results filtering methods."""

    def test_filter_by_threshold(self):
        """Test filtering results by confidence threshold."""
        from easy_rfdetr.results import Results

        results = Results(
            boxes=torch.tensor([[10, 10, 50, 50], [100, 100, 200, 200]]),
            scores=torch.tensor([0.9, 0.3]),
            class_ids=torch.tensor([0, 1]),
            class_names=["person", "car"],
        )

        filtered = results.filter(threshold=0.5)

        assert len(filtered.boxes) == 1
        assert filtered.scores[0] == 0.9


class TestResultsSave:
    """Tests for Results save method."""

    def test_save_method_exists(self):
        """Test save method exists on Results."""
        from easy_rfdetr.results import Results

        results = Results(
            boxes=torch.tensor([]),
            scores=torch.tensor([]),
            class_ids=torch.tensor([]),
            class_names=[],
        )

        assert hasattr(results, "save")
        assert callable(results.save)

    def test_save_without_original_image_raises(self):
        """Test save raises error without original image."""
        from easy_rfdetr.results import Results

        results = Results(
            boxes=torch.tensor([[10, 10, 50, 50]]),
            scores=torch.tensor([0.9]),
            class_ids=torch.tensor([0]),
            class_names=["person"],
        )

        with pytest.raises(ValueError, match="No original image"):
            results.save("output.jpg")


class TestResultsShow:
    """Tests for Results show method."""

    def test_show_method_exists(self):
        """Test show method exists on Results."""
        from easy_rfdetr.results import Results

        results = Results(
            boxes=torch.tensor([]),
            scores=torch.tensor([]),
            class_ids=torch.tensor([]),
            class_names=[],
        )

        assert hasattr(results, "show")
        assert callable(results.show)
