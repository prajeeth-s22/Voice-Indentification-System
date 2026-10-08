"""
train_model.py
──────────────
Multi-model training pipeline for the Voice Language Identification System.

Usage:
    python train_model.py

Pipeline:
1. Reads all WAV files from dataset/<language>/ (English, Hindi, Tamil, Telugu).
2. Extracts 80-dimensional MFCC feature vectors (40 means + 40 standard deviations).
3. Uses the EXACT SAME train/test split (random_state=42) across ALL classifiers for fair comparison.
4. Trains 6 distinct classification models:
     - Random Forest
     - Support Vector Machine (SVM)
     - K-Nearest Neighbors (KNN)
     - Logistic Regression
     - Decision Tree
     - Gradient Boosting
5. Evaluates accuracy, precision, recall, F1-score, and confusion matrix for each model.
6. Dynamically identifies the best-performing model.
7. Saves all models, the label encoder, and a comprehensive comparison report into models/.
"""

import os
import sys
import json
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.audio_processing import load_audio, preprocess_audio
from utils.feature_extraction import extract_mfcc_features
from utils.model_registry import (
    MODEL_DEFINITIONS,
    get_model_path,
    clear_model_cache,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(BASE_DIR, "dataset")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
REPORT_PATH  = os.path.join(MODELS_DIR, "training_report.json")
LEGACY_MODEL_PATH = os.path.join(MODELS_DIR, "language_model.pkl")

LANGUAGES = ["english", "hindi", "tamil", "telugu"]
MIN_FILES_PER_CLASS = 2   # Warn if fewer than this


def collect_dataset():
    """
    Walk through dataset/<language>/ folders and extract MFCC features.
    Returns:
        X (np.ndarray): (n_samples, 80)
        y (np.ndarray): (n_samples,) string language labels
        file_counts (dict): counts per language
    """
    features_list = []
    labels_list   = []
    file_counts   = {}

    for lang in LANGUAGES:
        lang_dir = os.path.join(DATASET_DIR, lang)
        os.makedirs(lang_dir, exist_ok=True)

        wav_files = [
            f for f in os.listdir(lang_dir)
            if f.lower().endswith(".wav")
        ]
        file_counts[lang] = len(wav_files)

        if len(wav_files) < MIN_FILES_PER_CLASS:
            print(f"  [!]  '{lang}': only {len(wav_files)} file(s) found "
                  f"(need at least {MIN_FILES_PER_CLASS}). Skipping.")
            continue

        print(f"  [OK]  '{lang}': {len(wav_files)} file(s) found.")

        for wav_file in wav_files:
            filepath = os.path.join(lang_dir, wav_file)
            try:
                audio, sr = load_audio(filepath)
                audio     = preprocess_audio(audio, sr)
                feats     = extract_mfcc_features(audio, sr)
                features_list.append(feats)
                labels_list.append(lang)
            except Exception as e:
                print(f"    [X] Skipping {wav_file}: {e}")

    return np.array(features_list), np.array(labels_list), file_counts


def train_all_models(X, y):
    """
    Train and evaluate all 6 classifiers using the SAME train/test split.
    Returns:
        trained_models (dict): {model_id: fitted_classifier}
        le (LabelEncoder): fitted label encoder
        comparison_results (dict): evaluation metrics and reports for each model
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    unique_classes = np.unique(y_enc)
    if len(unique_classes) < 2:
        raise ValueError("Need samples from at least 2 languages to train.")

    # Stratified train/test split for fair evaluation
    test_size = 0.2 if len(X) >= 10 else 0.1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=42, stratify=y_enc
    )

    trained_models = {}
    models_evaluation = {}

    best_model_id = None
    best_acc = -1.0
    best_f1 = -1.0

    print(f"\n[2/4] Training {len(MODEL_DEFINITIONS)} classification models on identical features...")
    print(f"      Train samples: {len(X_train)}  |  Test samples: {len(X_test)}\n")

    for m_id, m_def in MODEL_DEFINITIONS.items():
        name = m_def["name"]
        print(f"  --> Training {name} ...", end=" ", flush=True)

        clf = m_def["create"]()
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = round(accuracy_score(y_test, y_pred) * 100, 2)

        report = classification_report(
            y_test, y_pred,
            target_names=le.classes_,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(y_test, y_pred).tolist()

        weighted = report.get("weighted avg", {})
        prec = round(weighted.get("precision", 0) * 100, 2)
        rec  = round(weighted.get("recall", 0) * 100, 2)
        f1   = round(weighted.get("f1-score", 0) * 100, 2)

        print(f"Done! Test Accuracy: {acc}% | F1: {f1}%")

        trained_models[m_id] = clf
        models_evaluation[m_id] = {
            "id": m_id,
            "name": name,
            "short_name": m_def.get("short_name", name),
            "description": m_def.get("description", ""),
            "filename": m_def["filename"],
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "report": report,
            "confusion_matrix": cm,
        }

        # Track best model (accuracy primary, F1 secondary)
        if (acc > best_acc) or (acc == best_acc and f1 > best_f1):
            best_acc = acc
            best_f1 = f1
            best_model_id = m_id

    # Fallback to random forest if no best found
    if not best_model_id:
        best_model_id = "random_forest"

    summary = {
        "models": models_evaluation,
        "best_model": best_model_id,
        "best_model_name": models_evaluation[best_model_id]["name"],
        "best_accuracy": models_evaluation[best_model_id]["accuracy"],
        # Top-level backwards compatibility fields for existing UI & routes:
        "accuracy": models_evaluation[best_model_id]["accuracy"],
        "report": models_evaluation[best_model_id]["report"],
        "confusion_matrix": models_evaluation[best_model_id]["confusion_matrix"],
        "classes": le.classes_.tolist(),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    return trained_models, le, summary


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("\n" + "=" * 65)
    print("  Voice Language Identification - Multi-Model Training Pipeline")
    print("=" * 65)

    # 1. Collect data
    print("\n[1/4] Scanning dataset folders ...")
    X, y, file_counts = collect_dataset()

    if len(X) == 0:
        print("\n[ERROR] No audio files found in dataset/ folders.")
        print("        Place WAV files in dataset/<language>/ and re-run.\n")
        sys.exit(1)

    print(f"\n      Total samples collected: {len(X)}")

    # 2. Train all models
    try:
        trained_models, le, summary = train_all_models(X, y)
    except ValueError as e:
        print(f"\n[ERROR] Training error: {e}\n")
        sys.exit(1)

    summary["file_counts"] = file_counts

    # 3. Save all models & artifacts
    print("\n[3/4] Saving models and evaluation reports ...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    for m_id, clf in trained_models.items():
        save_path = get_model_path(m_id)
        joblib.dump(clf, save_path)
        print(f"      Saved: {MODEL_DEFINITIONS[m_id]['name']:<28} -> models/{os.path.basename(save_path)}")

    # Preserve legacy language_model.pkl for Random Forest
    if "random_forest" in trained_models:
        joblib.dump(trained_models["random_forest"], LEGACY_MODEL_PATH)
        print(f"      Saved: {'Legacy Random Forest':<28} -> models/language_model.pkl")

    joblib.dump(le, ENCODER_PATH)
    print(f"      Saved: {'Label Encoder':<28} -> models/label_encoder.pkl")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"      Saved: {'Comprehensive Report':<28} -> models/training_report.json")

    # Clear model cache in memory
    clear_model_cache()

    # 4. Display Comparison Table
    print("\n[4/4] Model Comparison Table\n")
    header = f"{'Model':<24} {'Accuracy':>10} {'Precision':>11} {'Recall':>9} {'F1-Score':>10}"
    print(header)
    print("-" * len(header))
    for m_id, m_info in summary["models"].items():
        is_best = " (BEST)" if m_id == summary["best_model"] else ""
        print(
            f"{m_info['short_name'] + is_best:<24} "
            f"{m_info['accuracy']:>9.1f}% "
            f"{m_info['precision']:>10.1f}% "
            f"{m_info['recall']:>8.1f}% "
            f"{m_info['f1_score']:>9.1f}%"
        )
    print("-" * len(header))

    print(f"\n  ★ Best Performing Model : {summary['best_model_name']}")
    print(f"  ★ Test Set Accuracy     : {summary['best_accuracy']}%")

    print("\n[DONE] All 6 models trained, evaluated, and saved successfully!")
    print("       Run the web app with:  python app.py\n")


if __name__ == "__main__":
    main()
