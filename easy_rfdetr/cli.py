"""CLI for easy_rfdetr."""

import typer
from typing import Optional
from pathlib import Path

from rich.console import Console

app = typer.Typer(help="easy-rfdetr: YOLO-like API for RF-DETR")
console = Console()


@app.command()
def predict(
    source: str = typer.Argument(..., help="Image path, URL, or directory"),
    model: str = typer.Option(
        "medium", "--model", "-m", help="Model size: nano, small, medium, large"
    ),
    threshold: float = typer.Option(0.5, "--threshold", "-t", help="Confidence threshold"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    device: str = typer.Option("auto", "--device", "-d", help="Device: auto, cuda, cpu, mps"),
    show: bool = typer.Option(False, "--show", help="Display results"),
):
    """Run object detection on image(s)."""
    from easy_rfdetr import RFDETR

    console.print(f"[cyan]Loading {model} model...[/cyan]")
    detector = RFDETR(model, device=device)

    console.print(f"[cyan]Running detection on {source}...[/cyan]")
    results = detector.predict(source, threshold=threshold)

    if output:
        if isinstance(results, list):
            for i, r in enumerate(results):
                out_path = Path(output) / f"result_{i}.jpg"
                r.save(str(out_path))
        else:
            results.save(output)
    else:
        results.save("output.jpg")

    if show:
        results.show()

    console.print(f"[green]✅ Done! Detected {len(results.boxes)} objects[/green]")


@app.command()
def benchmark(
    model: str = typer.Option("medium", "--model", "-m", help="Model size"),
    iterations: int = typer.Option(100, "--iterations", "-n", help="Number of iterations"),
):
    """Run benchmark on model."""
    from easy_rfdetr import RFDETR

    detector = RFDETR(model)
    detector.benchmark(iterations=iterations)


@app.command()
def serve(
    model: str = typer.Option("medium", "--model", "-m", help="Model size"),
    port: int = typer.Option(7860, "--port", "-p", help="Port number"),
):
    """Launch Gradio web interface."""
    from easy_rfdetr import RFDETR

    detector = RFDETR(model)
    detector.ui()


def main():
    app()


if __name__ == "__main__":
    main()
