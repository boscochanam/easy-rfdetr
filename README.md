# easy-rfdetr

<p align="center">
  <img src="https://img.shields.io/pypi/v/easy-rfdetr" alt="PyPI Version">
  <img src="https://img.shields.io/pypi/dm/easy-rfdetr" alt="PyPI Downloads">
  <img src="https://img.shields.io/pypi/pyversions/easy-rfdetr" alt="Python Versions">
  <a href="https://colab.research.google.com/github/easy-rfdetr/easy-rfdetr/blob/main/examples/demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>
</p>

<!-- Add your demo GIF here: -->
<!-- ![RFDETR Demo](assets/demo.gif) -->

## 3 Lines to Detect Objects

```python
from easy_rfdetr import RFDETR
model = RFDETR("medium")
model("image.jpg").show()
```

That's it. No preprocessing, no tensor handling, just works.

---

## Common Workflows

| What you want | How to do it |
|--------------|--------------|
| Quick detection | `model("image.jpg").show()` |
| From URL | `model("https://example.com/img.jpg")` |
| Batch multiple | `model(["a.jpg", "b.jpg"])` |
| Higher accuracy | `model("large")` |
| Faster/Lighter | `model("nano")` |
| Adjust confidence | `model("img.jpg", threshold=0.8)` |
| Save output | `model("img.jpg").save("out.jpg")` |
| Get raw data | `r = model("img.jpg"); print(r.boxes, r.labels)` |
| Web interface | `model.ui()` |
| Speed test | `model.benchmark()` |

---

## Installation

```bash
pip install easy-rfdetr
```

With web UI support:
```bash
pip install easy-rfdetr[gradio]
```

---

## Why easy-rfdetr?

- **Transformer accuracy** - Better than YOLO on AP50 (73.6-75.1%)
- **YOLO-like speed** - Real-time on GPU (6-10ms on T4)
- **Auto device** - Uses CUDA, MPS (Mac), or CPU automatically
- **Human labels** - Returns "person" not "0"
- **Beautiful by default** - Neon colors, clean boxes

---

## Model Sizes

| Model | Speed | Accuracy | Best for |
|-------|-------|----------|----------|
| nano | ~2ms | 67.6% AP50 | Embedded/Edge |
| small | ~4ms | 72.1% AP50 | Fast prototyping |
| medium | ~6ms | 73.6% AP50 | **Default - balanced** |
| large | ~10ms | 75.1% AP50 | Maximum accuracy |

---

## CLI Usage

```bash
rfdetr predict source=image.jpg
rfdetr predict source=image.jpg --threshold 0.7
rfdetr benchmark --model medium
```

---

## Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- CUDA GPU or Apple Silicon (optional, falls back to CPU)

---

## License

Apache 2.0 - See [LICENSE](LICENSE)

---

Built on [RF-DETR](https://github.com/roboflow/rf-detr) by Roboflow.
