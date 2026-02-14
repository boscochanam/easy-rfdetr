"""Core RFDETR class for easy_rfdetr."""

import os
import time
from typing import Union, List, Optional, Dict, Any
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from rich.console import Console
from rich.table import Table

from easy_rfdetr.utils import (
    get_device,
    download_weights,
    load_image,
    letterbox,
    get_class_names,
    get_model_class,
    get_model_resolution,
)
from easy_rfdetr.results import Results
from easy_rfdetr.visualizer import annotate_image

console = Console()

MODEL_STATS = {
    "nano": {"params": "3.2M", "gflops": "~5", "ap50": "67.6%"},
    "small": {"params": "19.7M", "gflops": "~20", "ap50": "72.1%"},
    "medium": {"params": "33.9M", "gflops": "~40", "ap50": "73.6%"},
    "large": {"params": "52.0M", "gflops": "~80", "ap50": "75.1%"},
}


class RFDETR:
    """RFDETR - YOLO-like API for Real-Time Detection Transformer.

    A simplified interface for running object detection with RF-DETR models.

    Example:
        >>> from easy_rfdetr import RFDETR
        >>> model = RFDETR("medium")
        >>> results = model.predict("image.jpg")
        >>> results.show()
    """

    def __init__(
        self,
        model: str = "medium",
        weights: Optional[str] = None,
        device: str = "auto",
        verbose: bool = True,
        threshold: float = 0.5,
    ):
        """Initialize RFDETR model.

        Args:
            model: Model size (nano, small, medium, large) or config dict
            weights: Custom weights path (optional)
            device: Device to use (auto, cuda, cpu, mps)
            verbose: Print loading messages
            threshold: Default confidence threshold
        """
        self._model_name = model.lower() if isinstance(model, str) else "medium"
        self._weights_path = weights
        self._device = get_device(device)
        self._verbose = verbose
        self._threshold = threshold
        self._model = None
        self._rfdetr_model = None

        if self._verbose:
            console.print("[bold cyan]🚀 Loading RFDETR model...[/bold cyan]")

        self._load_model()

        if self._verbose:
            self._print_model_info()

    def _load_model(self):
        """Load the RF-DETR model from roboflow package."""
        try:
            from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
            from rfdetr import RFDETRXLarge, RFDETR2XLarge
        except ImportError:
            console.print("[bold red]❌ Error: rfdetr package not found[/bold red]")
            console.print("[yellow]Please install: pip install rfdetr[/yellow]")
            raise

        model_classes = {
            "nano": RFDETRNano,
            "small": RFDETRSmall,
            "medium": RFDETRMedium,
            "large": RFDETRLarge,
            "xlarge": RFDETRXLarge,
            "2xlarge": RFDETR2XLarge,
        }

        if self._weights_path:
            model_key = self._model_name
            if self._weights_path.endswith(".pt"):
                weights_to_use = self._weights_path
            else:
                model_key = self._weights_path
                weights_to_use = None
        else:
            model_key = self._model_name
            if model_key not in model_classes:
                model_key = "medium"
            weights_to_use = None

        model_class = model_classes.get(model_key, RFDETRMedium)

        try:
            self._model = model_class(weights=weights_to_use)
            self._model.to(self._device)
            self._model.eval()

            if self._verbose:
                console.print(f"[green]✅ Loaded model on {self._device}[/green]")

        except Exception as e:
            console.print(f"[bold red]❌ Failed to load model: {e}[/bold red]")
            raise

    def _print_model_info(self):
        """Print model information table."""
        stats = MODEL_STATS.get(self._model_name, MODEL_STATS["medium"])

        table = Table(title=f"RFDETR-{self._model_name.title()} Model Info")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Model", f"RFDETR-{self._model_name.title()}")
        table.add_row("Device", self._device)
        table.add_row("Params", stats.get("params", "N/A"))
        table.add_row("GFLOPs", stats.get("gflops", "N/A"))
        table.add_row("COCO AP50", stats.get("ap50", "N/A"))
        table.add_row("Threshold", str(self._threshold))

        console.print(table)

    @property
    def device(self) -> str:
        """Get current device."""
        return self._device

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name

    def predict(
        self,
        source: Union[str, Image.Image, np.ndarray, List],
        threshold: Optional[float] = None,
        verbose: bool = True,
    ) -> Union[Results, List[Results]]:
        """Run object detection on image(s).

        Args:
            source: Image path, URL, PIL Image, numpy array, or list
            threshold: Confidence threshold (uses default if None)
            verbose: Print progress messages

        Returns:
            Results object or list of Results
        """
        if threshold is None:
            threshold = self._threshold

        if isinstance(source, (list, tuple)):
            return self._predict_batch(source, threshold, verbose)

        return self._predict_single(source, threshold, verbose)

    def _predict_single(
        self,
        source: Union[str, Image.Image, np.ndarray],
        threshold: float,
        verbose: bool,
    ) -> Results:
        """Run prediction on single image."""
        if isinstance(source, str):
            if source.startswith("http"):
                import requests

                response = requests.get(source, timeout=10)
                response.raise_for_status()
                original_image = Image.open(response.raw).convert("RGB")
            else:
                original_image = Image.open(source).convert("RGB")
        elif isinstance(source, np.ndarray):
            original_image = Image.fromarray(source)
        elif isinstance(source, Image.Image):
            original_image = source.convert("RGB")
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        resolution = get_model_resolution(self._model_name)
        letterboxed, ratio, pad = letterbox(original_image, (resolution, resolution))

        img_array = np.array(letterboxed)

        with torch.no_grad():
            predictions = self._model.predict(
                img_array,
                conf=threshold,
            )

        if predictions is None or len(predictions) == 0:
            return Results(
                boxes=torch.tensor([]),
                scores=torch.tensor([]),
                class_ids=torch.tensor([]),
                class_names=[],
                original_image=original_image,
            )

        pred = predictions[0]

        boxes = pred.boxes.xyxy.cpu()
        scores = pred.boxes.conf.cpu()
        class_ids = pred.boxes.class_id.cpu()

        class_names_list = get_class_names()
        labels = [
            class_names_list[cid] if cid < len(class_names_list) else str(cid)
            for cid in class_ids
        ]

        return Results(
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            class_names=labels,
            original_image=original_image,
        )

    def _predict_batch(
        self,
        sources: List,
        threshold: float,
        verbose: bool,
    ) -> List[Results]:
        """Run prediction on batch of images."""
        results = []
        for i, source in enumerate(sources):
            if verbose:
                console.print(f"Processing {i + 1}/{len(sources)}...")
            results.append(self._predict_single(source, threshold, verbose))
        return results

    def benchmark(
        self,
        iterations: int = 100,
        image_size: Optional[tuple] = None,
    ) -> Dict[str, float]:
        """Run benchmark to measure inference speed.

        Args:
            iterations: Number of inference iterations
            image_size: Input image size (uses model default if None)

        Returns:
            Dictionary with timing metrics
        """
        if image_size is None:
            resolution = get_model_resolution(self._model_name)
            image_size = (resolution, resolution)

        dummy_image = Image.new("RGB", image_size)

        console.print(f"[cyan]🔧 Running benchmark ({iterations} iterations)...[/cyan]")

        warmup = 5
        for _ in range(warmup):
            _ = self._predict_single(dummy_image, self._threshold, False)

        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = self._predict_single(dummy_image, self._threshold, False)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        times = np.array(times)
        mean_time = np.mean(times)
        std_time = np.std(times)
        fps = 1000.0 / mean_time

        result = {
            "inference_time_ms": round(mean_time, 2),
            "std_time_ms": round(std_time, 2),
            "fps": round(fps, 2),
            "min_ms": round(np.min(times), 2),
            "max_ms": round(np.max(times), 2),
        }

        table = Table(title="Benchmark Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Mean Inference", f"{result['inference_time_ms']} ms")
        table.add_row("FPS", f"{result['fps']}")
        table.add_row("Min", f"{result['min_ms']} ms")
        table.add_row("Max", f"{result['max_ms']} ms")

        console.print(table)

        return result

    def ui(self, title: str = "RFDETR Demo"):
        """Launch Gradio web interface.

        Args:
            title: App title
        """
        try:
            import gradio as gr
        except ImportError:
            console.print(
                "[yellow]⚠️  Gradio not installed. Install with: pip install gradio[/yellow]"
            )
            return

        def predict_fn(image, threshold):
            if image is None:
                return None
            results = self._predict_single(image, threshold, False)
            if len(results.boxes) > 0:
                annotated = annotate_image(
                    np.array(results.original_image),
                    results.boxes,
                    results.labels,
                    results.scores,
                )
                return Image.fromarray(annotated)
            return results.original_image

        demo = gr.Interface(
            fn=predict_fn,
            inputs=[
                gr.Image(label="Input Image", type="pil"),
                gr.Slider(
                    0, 1, value=self._threshold, step=0.05, label="Confidence Threshold"
                ),
            ],
            outputs=gr.Image(label="Detections"),
            title=title,
            description="RFDETR: Real-Time Object Detection with Transformers",
        )

        console.print("[green]🚀 Launching Gradio interface...[/green]")
        demo.launch()

    def __call__(self, source, **kwargs):
        """Alias for predict method."""
        return self.predict(source, **kwargs)
