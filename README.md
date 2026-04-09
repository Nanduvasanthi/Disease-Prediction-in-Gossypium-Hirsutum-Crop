# 🌱 CottonCare AI — Disease Prediction in *Gossypium hirsutum*

> An end-to-end deep learning web application for automated cotton crop disease detection, Grad-CAM explainability, and AI-powered disease advisory.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Disease Classes](#-disease-classes)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Running the App](#-running-the-app)
- [Model Details](#-model-details)
- [Chatbot Architecture](#-chatbot-architecture)
- [Configuration](#-configuration)
- [Testing the Prediction Pipeline](#-testing-the-prediction-pipeline)
- [Dataset](#-dataset)
- [Team](#-team)

---

## 🔍 Overview

Cotton (*Gossypium hirsutum*) is one of India's most commercially important crops, but it is highly vulnerable to a range of diseases and pest infestations that can cause yield losses of up to 40%. Manual disease identification by farmers is slow, inconsistent, and requires expert knowledge that is not always accessible in rural areas.

**CottonCare AI** solves this by providing:
- Instant AI-powered disease classification from a single leaf photo
- Visual explanation of *why* the model made its prediction (Grad-CAM heatmaps)
- An interactive AI chatbot for disease management advice, backed by both a curated knowledge base and the Google Gemini API

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **Deep Learning Classification** | EfficientNetB3 transfer learning model with **98.22%** validation accuracy |
| 🔍 **Grad-CAM Explainability** | Heatmap overlay showing which leaf regions drove the prediction |
| 🛡️ **Invalid Image Detection** | Multi-factor analysis (green ratio, edge density, texture, color variance) to reject non-cotton images |
| 💬 **AI Chatbot** | Dual-tier system: local knowledge base first, Google Gemini 2.5 Flash API fallback |
| 📊 **Confidence Scoring** | Per-prediction confidence percentage with a visual progress bar |
| 🏷️ **Severity Badges** | Automatic Low / Medium / High severity classification per disease |
| 🌐 **Streamlit Web UI** | Fully interactive browser-based app — no installation required for end users |

---

## 🦠 Disease Classes

The model detects **8 categories** of cotton plant conditions:

| # | Class | Severity | Description |
|---|---|---|---|
| 1 | **Aphids** | Medium | Small sap-sucking insects causing leaf curling and yellowing |
| 2 | **Army Worm** | High | Caterpillar infestation causing chewed leaves and boll damage |
| 3 | **Bacterial Blight** | High | Water-soaked angular lesions turning brown/necrotic |
| 4 | **Cotton Boll Rot** | High | Fungal infection causing decay of developing cotton bolls |
| 5 | **Green Cotton Boll** | Low | Healthy developing cotton bolls (non-disease class) |
| 6 | **Healthy Leaf** | Low | Normal, disease-free cotton foliage |
| 7 | **Powdery Mildew** | Medium | White powdery fungal coating on leaf surfaces |
| 8 | **Target Spot** | Medium | Circular concentric ring lesions on leaves |

---

## 📁 Project Structure

```
Disease-Prediction-in-Gossypium-Hirsutum-Crop/
│
├── app.py                          # Main Streamlit web application
├── predict.py                      # Prediction pipeline + invalid image detection
├── gradcam.py                      # Grad-CAM heatmap generation
├── chatbot.py                      # AI chatbot (local KB + Gemini API)
├── disease_knowledge.py            # Structured local disease knowledge base
├── train_efficientnet.py           # Model training script (EfficientNetB3)
├── evaluate_model.py               # Model evaluation + confusion matrix
├── file_picker_component.py        # File upload UI helper component
│
├── efficientnet_cotton_disease.keras   # Trained EfficientNetB3 model
├── Cotton_plant_disease_v1_98_22.h5    # Baseline CNN model (98.22% accuracy)
│
├── requirements.txt                # Python dependencies
└── .gitignore
```

---

## ⚙️ How It Works

```
User uploads leaf image
        │
        ▼
┌─────────────────────────────┐
│   Invalid Image Detection   │  ← Green ratio, edge density,
│   (predict.py)              │    texture & color variance checks
└────────────┬────────────────┘
             │ Valid cotton leaf
             ▼
┌─────────────────────────────┐
│   EfficientNetB3 Model      │  ← Preprocessed 224×224 image
│   (efficientnet_cotton      │    EfficientNetV2 preprocessing
│    _disease.keras)          │
└────────────┬────────────────┘
             │ Class + Confidence
             ▼
┌─────────────────────────────┐
│   Grad-CAM Heatmap          │  ← Gradients of predicted class
│   (gradcam.py)              │    w.r.t. last conv layer output
└────────────┬────────────────┘
             │ Overlayed image
             ▼
┌─────────────────────────────┐
│   Streamlit UI (app.py)     │  ← Result + Severity + Confidence
│                             │    + Heatmap + Chatbot
└─────────────────────────────┘
             │ User asks a question
             ▼
┌─────────────────────────────┐
│   Chatbot (chatbot.py)      │  ← Tier 1: disease_knowledge.py
│                             │    Tier 2: Google Gemini 2.5 Flash
└─────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Web Framework** | Streamlit |
| **Deep Learning** | TensorFlow / Keras |
| **Model Architecture** | EfficientNetB3 (Transfer Learning) |
| **Explainability** | Grad-CAM (GradientTape) |
| **Image Processing** | OpenCV, Pillow, NumPy, SciPy |
| **AI Chatbot** | Google Gemini 2.5 Flash (`google-genai`) |
| **Visualization** | Matplotlib |
| **Language** | Python 3.10+ |

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Nanduvasanthi/Disease-Prediction-in-Gossypium-Hirsutum-Crop.git
cd Disease-Prediction-in-Gossypium-Hirsutum-Crop
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt includes:**
```
streamlit
tensorflow
numpy
opencv-python
Pillow
google-genai
matplotlib
scipy
```

### 4. Set up your Gemini API Key

The chatbot uses Google's Gemini API. Create a `.env` file based on the `env.example` and replace `YOUR_GEMINI_API_KEY_HERE` with your actual key:

```
GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE
```

> Get a free API key at [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

### 5. Ensure model files are present

Both model files must be in the **root project directory**:

```
efficientnet_cotton_disease.keras     ← Primary model used by predict.py
Cotton_plant_disease_v1_98_22.h5      ← Baseline CNN model
```

---

## 🚀 Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 🧠 Model Details

### EfficientNetB3 — Primary Model

| Property | Value |
|---|---|
| Architecture | EfficientNetB3 (Transfer Learning from ImageNet) |
| Input Size | 224 × 224 × 3 |
| Preprocessing | `EfficientNetV2` preprocess_input |
| Output Classes | 8 |
| Validation Accuracy | **98.22%** |
| Saved As | `efficientnet_cotton_disease.keras` |
| Confidence Threshold | 80% (configurable in `predict.py`) |

### Invalid Image Detection

Before running the model, `predict.py` applies a **4-factor leaf verification** to reject non-cotton images:

| Check | Method | Threshold |
|---|---|---|
| 🌿 Green pixel ratio | HSV multi-range masking | ≥ 0.12 |
| 🍃 Leaf edge density | Canny edge detection on green region | ≥ 0.015 |
| 📊 Texture variance | Sobel gradient magnitude variance | ≥ 0.5 |
| 🎨 Color variance | LAB color space channel variance | ≥ 0.05 |

At least **3 of 4 checks** must pass for the image to be forwarded to the model.

### Grad-CAM Explainability

- Automatically finds the **last convolutional layer** in the model
- Uses `tf.GradientTape` to compute class-specific gradients
- Applies **jet colormap**: 🔴 Red = high influence, 🔵 Blue = low influence
- Overlays heatmap on original image with `alpha = 0.5`

---

## 💬 Chatbot Architecture

The chatbot (`chatbot.py`) uses a **two-tier response system**:

```
User Question
      │
      ▼
Tier 1: disease_knowledge.py
  ├── Detects intent keyword: "cause" / "symptom" / "treat" / "prevent"
  └── Returns structured answer instantly (< 100ms)
      │
      │ (if intent not matched)
      ▼
Tier 2: Google Gemini 2.5 Flash API
  ├── Constructs prompt with: disease name + last 3 messages + user question
  └── Returns real-time AI-generated agricultural advisory
```

The local knowledge base (`disease_knowledge.py`) covers all 8 disease classes with structured entries for causes, symptoms, treatment, and prevention.

---

## ⚙️ Configuration

Key thresholds in `predict.py` can be tuned to adjust sensitivity:

```python
CONFIDENCE_THRESHOLD       = 80    # Min model confidence (%) to accept prediction
GREEN_RATIO_THRESHOLD      = 0.12  # Min green pixel ratio for leaf detection
EDGE_DENSITY_THRESHOLD     = 0.015 # Min edge density within green region
TEXTURE_VARIANCE_THRESHOLD = 0.5   # Min Sobel gradient variance
COLOR_VARIANCE_THRESHOLD   = 0.05  # Min LAB color channel variance
```

---

## 🧪 Testing the Prediction Pipeline

You can test the prediction module directly from the command line without running the full Streamlit app:

```bash
python predict.py path/to/your/cotton_leaf.jpg
```

This will print:
- All 4 leaf verification check results
- Raw model prediction scores for all 8 classes
- Final predicted disease and confidence
- Whether Grad-CAM was generated successfully

---

## 📂 Dataset

This project was trained on the **Customized Cotton Disease Dataset** published on Kaggle by [saeedazfar](https://www.kaggle.com/saeedazfar).

> 📎 **Dataset:** [Customized Cotton Disease Dataset](https://www.kaggle.com/datasets/saeedazfar/customized-cotton-disease-dataset)  
> 👤 **Author:** saeedazfar  
> 📅 **Published:** September 2023  
> 🌐 **Source:** Kaggle

The dataset covers cotton pests, leaf diseases, and boll conditions across the 8 classes used in this project. We are grateful to the dataset author for making this resource publicly available, which made the training of our models possible.

If you use this dataset in your own work, please credit the original author and link back to the Kaggle dataset page above.

---

## 👥 Team

Developed as a final year B.Tech project at **Vasireddy Venkatadri Institute of Technology (VVIT)**, Guntur, Andhra Pradesh.

| Name | Roll Number |
|---|---|
| M Darshini Sai | 22BQ1A05D2 |
| M Nandu Vasanthi | 22BQ1A05E0 |
| M YamunaBai | 22BQ1A05E5 |
| N D Sandeep | 22BQ1A05F0 |

**Guide:** Dr. N. Sri Hari, Professor, Department of CSE, VVIT

---

> *"Early detection saves crops. Smart farming for better yields."*