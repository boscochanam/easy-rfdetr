# easy-rfdetr

<p align="center">
  <img src="https://img.shields.io/pypi/v/easy-rfdetr" alt="PyPI Version">
  <img src="https://img.shields.io/pypi/dm/easy-rfdetr" alt="PyPI Downloads">
  <img src="https://img.shields.io/pypi/pyversions/easy-rfdetr" alt="Python Versions">
  <a href="https://colab.research.google.com/github/easy-rfdetr/easy-rfdetr/blob/main/examples/train.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>
</p>

## Train a Model

```python
from easy_rfdetr import RFDETR

model = RFDETR("medium")
model.train(data="my_dataset/", epochs=50)
```

## Run Inference

```python
model("image.jpg").show()
```

## Install

```bash
pip install easy-rfdetr
```

---

## Training

Drop your dataset in a folder. We auto-detect COCO or YOLO format:

```
dataset/
├── train/images/ + labels/
├── valid/images/ + labels/
└── test/images/ + labels/
```

Train:
```python
model.train(data="dataset/", epochs=50, batch=8)
```

Resume:
```python
model.train(data="dataset/", resume=True)
```

---

## Inference

```python
model = RFDETR("medium")  # nano, small, medium, large

# From file
model("photo.jpg").show()

# From URL
model("https://example.com/img.jpg")

# Batch
model(["a.jpg", "b.jpg"])

# Confidence threshold
model("img.jpg", threshold=0.8)

# Get boxes
r = model("img.jpg")
print(r.boxes)     # xyxy
print(r.scores)    # confidence
print(r.labels)    # ["person", "car", ...]

# Save
r.save("output.jpg")
```

---

## Web UI

```python
model.ui()
```

---

## CLI

```bash
rfdetr predict source=image.jpg
rfdetr train data=dataset/ epochs=50
```

---

## Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- CUDA or Apple Silicon (optional)

---

## License

Apache 2.0 - See [LICENSE](LICENSE)

---

Built on [RF-DETR](https://github.com/roboflow/rf-detr) by Roboflow.
