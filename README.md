# 🌿 Plant Disease Detector

Detect whether a plant leaf is **healthy or diseased** in real time, using a webcam or a photo from your gallery. Powered by deep learning (MobileNetV2 transfer learning) with **96.39% validation accuracy** across **38 classes**.

---

## 📌 About the project

Plant diseases cause major crop losses, and diagnosing them early matters. This project trains a convolutional neural network on ~70K leaf images to recognize the plant type and its condition, then wraps the trained model in a simple desktop app:

1. Choose **Upload Image** or **Use Webcam**
2. The model identifies the plant and its condition
3. The result is shown on screen: plant, status (Healthy / Diseased), condition and confidence

---

## 📊 Dataset

- **Source:** [New Plant Diseases Dataset (Kaggle)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Training images:** 70,295
- **Validation images:** 17,572
- **Classes:** 38 (plant type + condition)
- **Input size:** 160 × 160 px

### The 38 classes

| Plant | Conditions | # |
|---|---|:-:|
| 🍎 Apple | Apple scab, Black rot, Cedar apple rust, Healthy | 4 |
| 🫐 Blueberry | Healthy | 1 |
| 🍒 Cherry | Powdery mildew, Healthy | 2 |
| 🌽 Corn (maize) | Cercospora leaf spot / Gray leaf spot, Common rust, Northern leaf blight, Healthy | 4 |
| 🍇 Grape | Black rot, Esca (Black measles), Leaf blight (Isariopsis leaf spot), Healthy | 4 |
| 🍊 Orange | Haunglongbing (Citrus greening) | 1 |
| 🍑 Peach | Bacterial spot, Healthy | 2 |
| 🫑 Pepper (bell) | Bacterial spot, Healthy | 2 |
| 🥔 Potato | Early blight, Late blight, Healthy | 3 |
| 🍓 Raspberry | Healthy | 1 |
| 🌱 Soybean | Healthy | 1 |
| 🎃 Squash | Powdery mildew | 1 |
| 🍓 Strawberry | Leaf scorch, Healthy | 2 |
| 🍅 Tomato | Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Spider mites (two-spotted), Target spot, Yellow leaf curl virus, Mosaic virus, Healthy | 10 |
| | **Total** | **38** |

---

## 🧠 Model & training

The model uses **transfer learning** with MobileNetV2 pretrained on ImageNet, a lightweight architecture that runs fast even on CPU.

**Architecture**

```
Input (160×160×3)
  → Data augmentation (random flip, rotation, zoom)
  → MobileNetV2 preprocessing
  → MobileNetV2 backbone (ImageNet weights)
  → Global Average Pooling
  → Dropout (0.3)
  → Dense (38, softmax)
```

**Training was done in two stages:**

| Stage | What was trained | Epochs | Learning rate | Val. accuracy | Val. loss |
|---|---|:-:|:-:|:-:|:-:|
| 1. Feature extraction | Only the new classification head (backbone frozen) | 8 | 1e-3 | 92.25% | 0.2396 |
| 2. **Fine-tuning** | Top layers of MobileNetV2 (from layer 100 onward) | 5 | 1e-5 | **96.39%** | **0.1114** |

Fine-tuning the upper layers of the backbone with a very small learning rate improved validation accuracy by about **4 percentage points** and cut validation loss by more than half.

**Final result: 96.39% validation accuracy.**

---

## ✨ Features

- 📷 **Live webcam mode:** real-time diagnosis with an on-screen overlay (predicts every few frames for smooth video)
- 🖼️ **Image upload mode:** pick any leaf photo from your files/gallery
- 🌱 Recognizes **38 plant/condition classes**
- 🟢🔴 Clear status: **Healthy** (green) or **Diseased** (red)
- 📈 Shows the prediction **confidence** percentage
- ⚡ Lightweight model (~25 MB), runs on CPU

---

## 🚀 Getting started

### 1. Clone the repository

```bash
git clone https://github.com/kamandNajari/plant_disease_detector.git
cd plant_disease_detector
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Install requirements

```bash
pip install -r requirements.txt
pip install jupyter
```

On Linux you also need tkinter for the file-picker window:

```bash
sudo apt install python3-tk
```

### 4. Run

```bash
jupyter notebook
```

Open `predict.ipynb` **from the repository root**, run the cell, then choose:

- **Upload Image:** select a leaf photo and see the result
- **Use Webcam:** live diagnosis (press `q` to quit)

> ⚠️ A desktop environment is required (the app uses tkinter and OpenCV windows). It will not work on Google Colab or headless servers.

---

## 📦 Requirements

```
tensorflow
opencv-python
numpy
```

Python 3.9+ is recommended.

---

## 🗂️ Project structure

```
plant_disease_detector/
├── train.ipynb                          # Model training (Google Colab, GPU)
├── predict.ipynb                        # Inference app (webcam / image upload)
├── requirements.txt
├── LICENSE
└── models/
    ├── plant_disease_classifier.keras   # Trained model
    └── plant_class_names.json           # Class labels
```

## 🔁 Retraining

Open `train.ipynb` in Google Colab with a GPU runtime, upload your Kaggle API key (`kaggle.json`), and run all cells. At the end, the trained model and class names are exported; place them in the `models/` folder.

---

## ⚠️ Limitations

- The dataset consists mostly of clean, single-leaf photos on simple backgrounds. Accuracy on real-world field photos (cluttered backgrounds, different lighting) may be lower.
- The model only knows the 38 classes listed above; other plants or diseases will be misclassified.
- This is an educational project and **not** a substitute for professional agricultural diagnosis.

---

## 📄 License

Released under the [MIT License](LICENSE).

## 👤 Author

**Kamand Najari** · [GitHub](https://github.com/kamandNajari)

Thanks for checking out this project💙✨️
