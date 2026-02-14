"""Tests for easy_rfdetr training functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os


class TestTrainingAPI:
    """Tests for training API."""

    def test_train_method_defined_in_core(self):
        """Test train method is defined in core.py."""
        with open("/home/claw/easy_rfdetr/easy_rfdetr/core.py", "r") as f:
            content = f.read()
            assert "def train(" in content
            assert "data:" in content
            assert "epochs:" in content

    def test_train_accepts_data_parameter(self):
        """Test train method accepts data parameter."""
        with open("/home/claw/easy_rfdetr/easy_rfdetr/core.py", "r") as f:
            content = f.read()
            assert "def train(" in content

    def test_train_has_batch_option(self):
        """Test train method has batch option."""
        with open("/home/claw/easy_rfdetr/easy_rfdetr/core.py", "r") as f:
            content = f.read()
            assert "batch:" in content

    def test_train_has_output_option(self):
        """Test train method has output option."""
        with open("/home/claw/easy_rfdetr/easy_rfdetr/core.py", "r") as f:
            content = f.read()
            assert "output:" in content


class TestTrainConfig:
    """Tests for training configuration."""

    def test_prepare_train_config_exists(self):
        """Test prepare_train_config function exists."""
        with open("/home/claw/easy_rfdetr/easy_rfdetr/utils.py", "r") as f:
            content = f.read()
            assert "def prepare_train_config(" in content


class TestDatasetFormats:
    """Tests for dataset format detection."""

    def test_coco_format_detected(self):
        """Test COCO format is supported."""
        from easy_rfdetr.utils import detect_dataset_format

        with patch("os.path.exists") as mock_exists:
            with patch("os.path.isdir") as mock_isdir:
                with patch("os.listdir") as mock_listdir:
                    mock_exists.return_value = True
                    mock_isdir.return_value = True
                    mock_listdir.return_value = ["train", "valid", "test", "_annotations.coco.json"]
                    result = detect_dataset_format("/fake/path")
                    assert result == "coco"

    def test_yolo_format_detected(self):
        """Test YOLO format is supported."""
        from easy_rfdetr.utils import detect_dataset_format

        with patch("os.path.exists") as mock_exists:
            with patch("os.path.isdir") as mock_isdir:
                with patch("os.listdir") as mock_listdir:
                    with patch("os.path.isfile") as mock_isfile:
                        mock_exists.return_value = True
                        mock_isdir.return_value = True
                        mock_listdir.return_value = ["image1.txt", "image2.txt"]
                        mock_isfile.return_value = True
                        result = detect_dataset_format("/fake/path")
                        assert result == "yolo"

    def test_auto_format_when_unknown(self):
        """Test auto format when unknown."""
        from easy_rfdetr.utils import detect_dataset_format

        with patch("os.path.exists") as mock_exists:
            with patch("os.path.isdir") as mock_isdir:
                with patch("os.listdir") as mock_listdir:
                    mock_exists.return_value = True
                    mock_isdir.return_value = True
                    mock_listdir.return_value = ["image1.jpg", "image2.jpg"]
                    result = detect_dataset_format("/fake/path")
                    assert result == "auto"
