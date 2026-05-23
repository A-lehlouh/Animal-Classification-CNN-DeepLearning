#  Animal Image Classification using CNN & Transfer Learning

A deep learning project that classifies images of animals (cats, dogs, and snakes) using Convolutional Neural Networks (CNN). The project compares four model variants — two custom CNN architectures trained from scratch vs. the same architectures using MobileNetV2 transfer learning.

---

## Dataset

- **Source:** Kaggle - Animal Dataset
- **Classes:** cats, dogs, snakes (1,000 images each)
- **Total images:** 3,000
- **Image size:** resized to 300×300

**Download via kagglehub:**
```python
import kagglehub
path = kagglehub.dataset_download("aboodlahloh/animal-dataset")
print("Path to dataset files:", path)
```

Or download directly from:
 https://www.kaggle.com/datasets/aboodlahloh/animal-dataset

---

##  Project Pipeline

1. **Load & Extract Dataset** — unzip and load images from `Animals/` folder
2. **Visualize Samples** — display a 4×4 grid of sample images per class
3. **Data Augmentation** — rotation, shift, shear, zoom, horizontal flip
4. **Data Generators** — `ImageDataGenerator` for train and validation sets
5. **Model Building** — two custom CNN architectures (A and B)
6. **Transfer Learning** — MobileNetV2 as frozen base for both architectures
7. **Training** — EarlyStopping + ModelCheckpoint callbacks
8. **Evaluation** — accuracy/loss curves plotted for all 4 models

---

##  Models

| Model | Architecture | Pretrained Weights |
|-------|-------------|-------------------|
| **Model_A** | Conv2D(16) → MaxPool → Conv2D(32) → MaxPool → Flatten → Dense(64) → Dense(3) | None (from scratch) |
| **Model_B** | Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool → Flatten → Dense(256) → Dense(64) → Dense(3) | None (from scratch) |
| **Model_A_TL** | MobileNetV2 (frozen) → Flatten → Dense(64) → Dense(3) | ImageNet |
| **Model_B_TL** | MobileNetV2 (frozen) → Flatten → Dense(64) → Dense(3) | ImageNet |

All models use `softmax` output, `categorical_crossentropy` loss, and `RMSprop` optimizer.

---

##  Training Details

- **Epochs:** up to 15 (with EarlyStopping, patience = 5)
- **Batch size:** 32
- **Input size:** 300 × 300 × 3
- **Callbacks:** EarlyStopping, ModelCheckpoint (saves best model as `.h5`)

---

##  Evaluation

Training and validation accuracy/loss curves are plotted for each model using a custom `history_plot()` function.

---

##  How to Run

1. Open the notebook in **Google Colab**
2. Install kagglehub and download the dataset:
```python
import kagglehub
path = kagglehub.dataset_download("aboodlahloh/animal-dataset")
```
3. Update the dataset path in the notebook if needed
4. Run all cells in order

---

##  Requirements

- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- kagglehub
- Google Colab (recommended)

---

##  Repository Structure

```
├── DeepLGPr1.ipynb           # Main notebook
└── README.md
```
