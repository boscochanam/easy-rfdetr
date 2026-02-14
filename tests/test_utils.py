"""Tests for easy_rfdetr utils module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys


class TestDeviceDetection:
    """Tests for device detection functionality."""

    def test_get_device_cpu_only(self):
        """Test get_device returns cpu when no CUDA."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch("platform.system", return_value="Linux"):
                from easy_rfdetr.utils import get_device

                device = get_device()
                assert device in ["cpu", "mps"]

    @patch("torch.cuda.is_available")
    def test_get_device_cuda_available(self, mock_cuda):
        """Test get_device returns cuda when available."""
        mock_cuda.return_value = True
        from easy_rfdetr.utils import get_device

        device = get_device()
        assert device == "cuda"

    @patch("torch.cuda.is_available")
    @patch("platform.system")
    @patch("torch.backends.mps.is_available")
    def test_get_device_mps_mac(self, mock_mps, mock_system, mock_cuda):
        """Test get_device returns mps on Mac."""
        mock_cuda.return_value = False
        mock_system.return_value = "Darwin"
        mock_mps.return_value = True
        from easy_rfdetr.utils import get_device
        import importlib
        import easy_rfdetr.utils

        importlib.reload(easy_rfdetr.utils)
        from easy_rfdetr.utils import get_device

        device = get_device()
        assert device == "mps"


class TestWeightDownload:
    """Tests for weight downloading functionality."""

    def test_download_weights_function_exists(self):
        """Test download_weights function exists."""
        from easy_rfdetr.utils import download_weights

        assert download_weights is not None

    @patch("requests.get")
    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_download_weights_nano(self, mock_makedirs, mock_exists, mock_get):
        """Test download_weights for nano model."""
        from easy_rfdetr.utils import download_weights

        mock_exists.return_value = False
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "100"}
        mock_response.iter_content = Mock(return_value=[b"test"])
        mock_get.return_value = mock_response

        result = download_weights("nano")
        assert result is not None

    @patch("os.path.exists")
    def test_download_weights_skip_existing(self, mock_exists):
        """Test download_weights skips if file exists."""
        from easy_rfdetr.utils import download_weights

        mock_exists.return_value = True

        result = download_weights("nano")
        assert result is not None


class TestInputHandling:
    """Tests for input handling functionality."""

    @patch("os.path.isfile")
    @patch("PIL.Image.open")
    def test_load_image_from_path(self, mock_open, mock_isfile):
        """Test loading image from file path."""
        from easy_rfdetr.utils import load_image

        mock_isfile.return_value = True
        mock_img = Mock()
        mock_img.convert = Mock(return_value=mock_img)
        mock_open.return_value = mock_img

        result = load_image("test.jpg")
        assert result is not None

    def test_load_image_from_url(self):
        """Test loading image from URL."""
        from easy_rfdetr.utils import load_image

        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.raw = Mock()
            mock_response.raw.read = Mock()
            mock_get.return_value = mock_response

            with patch("PIL.Image.open") as mock_open:
                mock_img = Mock()
                mock_img.convert = Mock(return_value=mock_img)
                mock_open.return_value = mock_img

                result = load_image("http://example.com/image.jpg")
                assert result is not None

    def test_load_image_from_pil(self):
        """Test load_image accepts PIL Image."""
        from easy_rfdetr.utils import load_image
        from PIL import Image

        img = Image.new("RGB", (100, 100))
        result = load_image(img)

        assert result is not None

    def test_load_image_from_numpy(self):
        """Test load_image accepts numpy array."""
        from easy_rfdetr.utils import load_image
        import numpy as np

        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        result = load_image(arr)

        assert result is not None


class TestLetterbox:
    """Tests for letterbox preprocessing."""

    def test_letterbox_function_exists(self):
        """Test letterbox function exists."""
        from easy_rfdetr.utils import letterbox

        assert letterbox is not None

    def test_letterbox_preserves_aspect_ratio(self):
        """Test letterbox preserves aspect ratio."""
        from easy_rfdetr.utils import letterbox
        from PIL import Image

        img = Image.new("RGB", (640, 480))
        result, ratio, pad = letterbox(img, (640, 640))

        assert ratio[0] == ratio[1]

    def test_letterbox_returns_tuple(self):
        """Test letterbox returns (image, ratio, padding)."""
        from easy_rfdetr.utils import letterbox
        from PIL import Image

        img = Image.new("RGB", (100, 100))
        result = letterbox(img, (640, 640))

        assert len(result) == 3


class TestClassNames:
    """Tests for class name handling."""

    def test_get_class_names_function_exists(self):
        """Test get_class_names function exists."""
        from easy_rfdetr.utils import get_class_names

        assert get_class_names is not None

    def test_get_coco_class_names(self):
        """Test COCO class names are loaded."""
        from easy_rfdetr.utils import get_class_names

        names = get_class_names()
        assert len(names) > 0
        assert "person" in names
