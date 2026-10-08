# Voice Language Identification System

**Speech & Language Processing Project**

---

## Overview

This system identifies the spoken language from audio recordings using two complementary recognition engines:

1. **🌲 Multi-Model Classical ML Suite (MFCC Features + Model Selection Layer)**
   - Extracts 80-dimensional MFCC feature vectors (40 means + 40 standard deviations) from audio clips.
   - Evaluates **6 distinct scikit-learn classifiers** on identical features with a fixed stratified train/test split:
     1. **Random Forest** (`RandomForestClassifier(n_estimators=100, random_state=42)`)
     2. **Support Vector Machine (SVM)** (`SVC(kernel="rbf", probability=True, random_state=42)`)
     3. **K-Nearest Neighbors (KNN)** (`KNeighborsClassifier(n_neighbors=5)`)
     4. **Logistic Regression** (`LogisticRegression(max_iter=1000, random_state=42)`)
     5. **Decision Tree** (`DecisionTreeClassifier(random_state=42)`)
     6. **Gradient Boosting** (`GradientBoostingClassifier(random_state=42)`)
   - Includes an **Auto – Best Model** option that dynamically uses the classifier with the highest test accuracy.
   - Provides full **Model Comparison Table** with Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

2. **✨ Google Gemini Multimodal Audio AI (Cloud Real-World Audio)**
   - Directly analyzes human speech across English, Hindi, Tamil, Telugu, and dozens of world languages.
   - Produces native script transcriptions, English translations, and phonetic reasoning.

---

## Pipeline

```text
Voice Input (Upload / Microphone Recording)
                    ↓
        Audio Preprocessing
 (Convert to Mono · Resample to 16,000 Hz · Amplitude Normalization · Silence Trimming)
                    ↓
       MFCC Feature Extraction
 (40 MFCCs → 40 Means + 40 Std Deviations = 80-dimensional feature vector)
                    ↓
         MODEL SELECTION LAYER
  [ Random Forest | SVM | KNN | Logistic Regression | Decision Tree | Gradient Boosting | Auto ]
                    ↓
      Selected Classification Model
                    ↓
            Language Prediction
                    ↓
             Confidence Score
                    ↓
          Existing Results Dashboard
```

---

## Supported Models & Baseline Hyperparameters

| Model | Scikit-Learn Class | Key Hyperparameters |
|---|---|---|
| **Random Forest** | `RandomForestClassifier` | `n_estimators=100, random_state=42, n_jobs=-1` |
| **SVM** | `SVC` | `kernel="rbf", probability=True, random_state=42` |
| **KNN** | `KNeighborsClassifier` | `n_neighbors=5` |
| **Logistic Regression** | `LogisticRegression` | `max_iter=1000, random_state=42` |
| **Decision Tree** | `DecisionTreeClassifier` | `random_state=42` |
| **Gradient Boosting** | `GradientBoostingClassifier` | `random_state=42` |
| **Auto – Best Model** | Dynamic Dispatch | Automatically selects model with highest test accuracy |

---

## Setup & Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **(Optional) Configure Gemini API Key:**
   Get a free key from [Google AI Studio](https://aistudio.google.com/app/apikey) and set it in your `.env` file or via the Web UI:
   ```bash
   GEMINI_API_KEY="your_api_key_here"
   ```

---

## Quick Start

### 1. Launch the Web Application
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser.

- Select **🌲 Local ML Classifiers** or **✨ Google Gemini AI**.
- Choose your preferred model from the **Classification Model** dropdown (or select **Auto – Best Model**).
- Upload an audio file or click **Record Voice Live**.
- Click **Identify Language & Classify**.
- View the predicted language, confidence score, and language probability distribution.

---

### 2. Multi-Model Training Pipeline

To train and evaluate all 6 models on your dataset:
```bash
python train_model.py
```

The script will:
1. Scan `dataset/english/`, `dataset/hindi/`, `dataset/tamil/`, `dataset/telugu/`.
2. Extract 80-dim MFCC features.
3. Train all 6 classifiers on the identical train/test split.
4. Calculate Accuracy, Precision, Recall, F1-score, and Confusion Matrices.
5. Identify the best-performing model dynamically.
6. Save all `.pkl` models and `training_report.json` in `models/`.

---

### 3. Command-Line Predictions

```bash
# Predict using SVM
python predict.py "path/to/audio.wav" --model svm

# Predict using Random Forest
python predict.py "path/to/audio.wav" --model random_forest

# Predict using Auto (dynamically chooses best model)
python predict.py "path/to/audio.wav" --model auto

# Predict using Google Gemini AI
python predict.py "path/to/audio.wav" --engine gemini --api-key YOUR_KEY
```

---

## Project Structure

```text
voice-language-identification/
│
├── app.py                     # Flask web server (Model selector, dual engine & recording)
├── predict.py                 # Multi-model prediction module (CLI & API)
├── train_model.py             # Multi-model training and evaluation pipeline
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── .env.example               # Template for API key configuration
│
├── utils/
│   ├── model_registry.py      # Central registry for all 6 classifiers
│   ├── audio_processing.py    # Audio loading, resampling, and normalization
│   ├── feature_extraction.py  # MFCC feature extraction (80-dim vector)
│   └── gemini_service.py      # Google Gemini multimodal audio recognition
│
├── dataset/                   # Dataset directory
│   ├── english/
│   ├── hindi/
│   ├── tamil/
│   └── telugu/
│
├── models/                    # Trained model artifacts
│   ├── random_forest.pkl
│   ├── svm.pkl
│   ├── knn.pkl
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── gradient_boosting.pkl
│   ├── language_model.pkl     # Legacy Random Forest fallback
│   ├── label_encoder.pkl      # Sklearn LabelEncoder
│   └── training_report.json   # Multi-model comparison report
│
├── static/
│   └── styles.css             # UI styling & animations
└── templates/
    └── index.html             # Web application frontend with model selector
```
