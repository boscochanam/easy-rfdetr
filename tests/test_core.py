"""Tests for easy_rfdetr core module (RFDETR class)."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch


class TestRFDETRInit:
    """Tests for RFDETR class initialization."""

    def test_rfdetr_class_exists(self):
        """Test that RFDETR class can be imported."""
        from easy_rfdetr import RFDETR

        assert RFDETR is not None

    def test_rfdetr_is_callable(self):
        """Test RFDETR is a class (callable)."""
        from easy_rfdetr import RFDETR

        assert callable(RFDETR)
