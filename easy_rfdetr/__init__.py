"""easy_rfdetr - YOLO-like API for RF-DETR object detection."""

import os
import sys
from rich.console import Console

ASCII_LOGO = """
[bold cyan]
 ██████╗ ███████╗███╗   ███╗ █████╗ ██╗    ██╗
██╔════╝ ██╔════╝████╗ ████║██╔══██╗██║    ██║
██║  ███╗█████╗  ██╔████╔██║███████║██║ █╗ ██║
██║   ██║██╔══╝  ██║╚██╔╝██║██╔══██║██║███╗██║
╚██████╔╝███████╗██║ ╚═╝ ██║██║  ██║╚███╔███╔╝
 ╚═════╝ ╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝ ╚══╝══╝ 
[/bold cyan]
[bold magenta]Rapid Detection Transformer - YOLO-like Ease[/bold magenta]
"""

console = Console()


def _print_logo():
    """Print ASCII logo on import."""
    if not sys.stdout.isatty():
        return
    try:
        console.print(ASCII_LOGO)
    except Exception:
        pass


_VERBOSE = os.environ.get("EASY_RFDETR_VERBOSE", "true").lower() == "true"

if _VERBOSE:
    _print_logo()

from easy_rfdetr.core import RFDETR
from easy_rfdetr.results import Results

__version__ = "0.1.0"
__author__ = "easy-rfdetr"

__all__ = ["RFDETR", "Results"]
