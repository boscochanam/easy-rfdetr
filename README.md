# easy-rfdetr

<p align="center">
  <img src="https://img.shields.io/pypi/v/easy-rfdetr" alt="PyPI Version">
  <img src="https://img.shields.io/pypi/dm/easy-rfdetr" alt="PyPI Downloads">
  <img src="https://img.shields.io/pypi/pyversions/easy-rfdetr" alt="Python Versions">
  <a href="https://colab.research.google.com/github/easy-rfdetr/easy-rfdetr/blob/main/examples/train.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>
</p>

<!-- Add your demo GIF here: -->
<!-- ![RFDETR Demo](assets/demo.gif) -->

## Train Custom Models in 3 Lines

```python
from easy_rfdetr import RFDETR

model = RFDETR("medium")
model.train(data="my_dataset/", epochs=50)
```

That's it. No config files, no CLI commands, no boilerplate.

---

## Why easy-rfdetr?

| What | rfdetr (original) | easy-rfdetr |
|------|-------------------|-------------|
| Training | 50+ lines of config | 1 line |
| Inference | Manual preprocessing | Works out of the box |
| Dataset format | Manual setup | Auto-detects COCO/YOLO |
| Device | Manual cuda/cpu | Auto-detects |

**Transformer accuracy. YOLO simplicity.**

---

## Common Workflows

| What you want | How to do it |
|--------------|--------------|
| **Train your model** | `model.train(data="data/", epochs=50)` |
| Continue training | `model.train(data="data/", resume=True)` |
| Use trained model | `model("image.jpg").show()` |
| From URL | `model("https://example.com/img.jpg")` |
| Batch predict | `model(["a.jpg", "b.jpg"])` |
| Adjust confidence | `model("img.jpg", threshold=0.8)` |
| Save output | `model("img.jpg").save("out.jpg")` |
| Get raw boxes | `r = model("img.jpg"); print(r.boxes)` |
| Web UI | `model.ui()` |
| Speed test | `model.benchmark()` |

---

## Installation

```bash
pip install easy-rfdetr
```

With web UI:
```bash
pip install easy-rfdetr[gradio]
```

---

## Training

### Dataset Format

Just drop your data in a folder. We auto-detect COCO or YOLO format:

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### Training Options

```python
model.train(
    data="my_dataset/",    # Required: path to dataset
    epochs=50,             # Default: 100
    batch=8,               # Default: 4
    lr=1e-4,              # Default: 0.0001
    output="runs/train",  # Default: ./runs/train
    imgsz=640,            # Optional: image size
    resume=False,          # Optional: resume from checkpoint
)
```

---

## Inference

```python
from easy_rfdetr import RFDETR

model = RFDETR("medium")           # nano, small, medium, large
results = model("image.jpg")

print(results.boxes)    # xyxy coordinates
print(results.scores)   # confidence scores
print(results.labels)   # ["person", "car", ...]

results.show()          # Display
results.save("out.jpg") # Save
```

---

## Model Sizes

| Model | Speed | Accuracy | Best for |
|-------|-------|----------|----------|
| nano | ~2ms | 67.6% AP50 | Embedded |
| small | ~4ms | 72.1% AP50 | Fast |
| **medium** | ~6ms | 73.6% AP50 | **Default** |
| large | ~10ms | 75.1% AP50 | Accuracy |

---

## Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- CUDA GPU or Apple Silicon (optional)

---

## License

Apache 2.0 - See [LICENSE](LICENSE)

---

Built on [RF-DETR](https://github.com/roboflow/rf-detr) by Roboflow.
