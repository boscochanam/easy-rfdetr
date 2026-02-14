"""Utility functions for easy_rfdetr."""

import os
import sys
import platform
import hashlib
from pathlib import Path
from typing import Union, Tuple, Optional, List

import torch
import numpy as np
from PIL import Image
import requests
from tqdm import tqdm


MODEL_CONFIG = {
    "nano": {"class": "RFDETRNano", "resolution": 384},
    "small": {"class": "RFDETRSmall", "resolution": 512},
    "medium": {"class": "RFDETRMedium", "resolution": 576},
    "large": {"class": "RFDETRLarge", "resolution": 704},
    "xlarge": {"class": "RFDETRXLarge", "resolution": 700},
    "2xlarge": {"class": "RFDETR2XLarge", "resolution": 880},
}

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

WEIGHTS_URLS = {
    "nano": "https://huggingface.co/rfdetr/rfdetr-nano/resolve/main/rfdetr_nano.pt",
    "small": "https://huggingface.co/rfdetr/rfdetr-small/resolve/main/rfdetr_small.pt",
    "medium": "https://huggingface.co/rfdetr/rfdetr-medium/resolve/main/rfdetr_medium.pt",
    "large": "https://huggingface.co/rfdetr/rfdetr-large/resolve/main/rfdetr_large.pt",
}

DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/easy_rfdetr")


def get_device(device: str = "auto") -> str:
    """Get the appropriate device for inference.

    Args:
        device: "auto", "cuda", "cpu", or "mps"

    Returns:
        Device string: "cuda", "mps", or "cpu"
    """
    if device != "auto":
        return device

    if torch.cuda.is_available():
        return "cuda"
    elif (
        platform.system() == "Darwin"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"
    else:
        return "cpu"


def get_cache_dir() -> Path:
    """Get or create the cache directory for weights."""
    cache_dir = Path(DEFAULT_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_weights(model_name: str, cache_dir: Optional[Path] = None) -> str:
    """Download model weights from HuggingFace.

    Args:
        model_name: Model size (nano, small, medium, large)
        cache_dir: Custom cache directory

    Returns:
        Path to downloaded weights file
    """
    if cache_dir is None:
        cache_dir = get_cache_dir()

    model_key = model_name.lower()
    if model_key not in WEIGHTS_URLS:
        model_key = "medium"

    weights_filename = f"rfdetr_{model_key}.pt"
    weights_path = cache_dir / weights_filename

    if weights_path.exists():
        return str(weights_path)

    url = WEIGHTS_URLS.get(model_key, WEIGHTS_URLS["medium"])
    print(f"⬇️  Downloading {model_name} weights from HuggingFace...")

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 8192

        with (
            open(weights_path, "wb") as f,
            tqdm(total=total_size, unit="B", unit_scale=True, desc=weights_filename) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"✅ Downloaded weights to {weights_path}")
        return str(weights_path)

    except Exception as e:
        print(f"⚠️  Download failed: {e}")
        raise


def load_image(source: Union[str, Image.Image, np.ndarray]) -> Image.Image:
    """Load an image from various sources.

    Args:
        source: File path, URL, PIL Image, or numpy array

    Returns:
        PIL Image
    """
    if isinstance(source, Image.Image):
        return source.convert("RGB")

    if isinstance(source, np.ndarray):
        return Image.fromarray(source)

    if isinstance(source, str):
        if source.startswith(("http://", "https://")):
            response = requests.get(source, timeout=10)
            response.raise_for_status()
            return Image.open(response.raw).convert("RGB")

        if os.path.isfile(source):
            return Image.open(source).convert("RGB")

        if os.path.isdir(source):
            raise ValueError("Use predict() with a directory to process multiple images")

    raise ValueError(f"Unsupported image source: {type(source)}")


def letterbox(
    image: Image.Image,
    target_size: Tuple[int, int],
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[Image.Image, Tuple[float, float], Tuple[int, int]]:
    """Resize image with letterboxing to maintain aspect ratio.

    Args:
        image: PIL Image
        target_size: Target (width, height)
        color: Padding color

    Returns:
        Tuple of (resized_image, scale_ratio, padding)
    """
    w, h = image.size
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = image.resize((new_w, new_h), Image.BICUBIC)

    result = Image.new("RGB", target_size, color)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    result.paste(resized, (paste_x, paste_y))

    return result, (scale, scale), (paste_x, paste_y)


def get_class_names() -> List[str]:
    """Get COCO class names."""
    return COCO_CLASSES


def get_model_class(model_name: str) -> str:
    """Get the Roboflow RF-DETR model class name.

    Args:
        model_name: nano, small, medium, large, etc.

    Returns:
        Model class name (e.g., "RFDETRMedium")
    """
    key = model_name.lower()
    if key not in MODEL_CONFIG:
        key = "medium"
    return MODEL_CONFIG[key]["class"]


def get_model_resolution(model_name: str) -> int:
    """Get the model input resolution.

    Args:
        model_name: nano, small, medium, large, etc.

    Returns:
        Input resolution (e.g., 576)
    """
    key = model_name.lower()
    if key not in MODEL_CONFIG:
        key = "medium"
    return MODEL_CONFIG[key]["resolution"]


def detect_dataset_format(dataset_path: str) -> str:
    """Detect dataset format (COCO or YOLO).

    Args:
        dataset_path: Path to dataset directory

    Returns:
        "coco", "yolo", or "auto"
    """
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path does not exist: {dataset_path}")

    train_dir = os.path.join(dataset_path, "train")
    if not os.path.isdir(train_dir):
        train_dir = dataset_path

    files = os.listdir(train_dir)

    if "_annotations.coco.json" in files or "_annotations_.coco.json" in files:
        return "coco"

    if any(f.endswith(".txt") for f in files if os.path.isfile(os.path.join(train_dir, f))):
        return "yolo"

    return "auto"


def prepare_train_config(
    data: str,
    epochs: int = 100,
    batch: int = 4,
    imgsz: int = None,
    lr: float = 1e-4,
    output: str = "runs/train",
    resume: bool = False,
    device: str = "auto",
    **kwargs,
) -> dict:
    """Prepare training configuration for rfdetr.

    Args:
        data: Path to dataset directory
        epochs: Number of training epochs
        batch: Batch size
        imgsz: Image size (uses model default if None)
        lr: Learning rate
        output: Output directory
        resume: Resume from checkpoint
        device: Device to use
        **kwargs: Additional training options

    Returns:
        Training configuration dict
    """
    config = {
        "dataset_dir": data,
        "epochs": epochs,
        "batch_size": batch,
        "lr": lr,
        "output_dir": output,
        "resume": resume if isinstance(resume, str) else None,
        "device": get_device(device),
    }

    if imgsz is not None:
        config["imgsz"] = imgsz

    config.update(kwargs)

    return config
