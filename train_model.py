"""
train_model.py
Training script for the Voice Language Identification System.

Usage:
    python train_model.py

Reads WAV files from dataset/<language>/, extracts MFCC features,
trains a Random Forest classifier, evaluates it, and saves the model.
"""

import os
import sys
import json
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
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

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH  = os.path.join(MODELS_DIR, "language_model.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
REPORT_PATH  = os.path.join(MODELS_DIR, "training_report.json")

LANGUAGES = ["english", "hindi", "tamil", "telugu"]
MIN_FILES_PER_CLASS = 2   # warn if fewer than this


# ── Helpers ────────────────────────────────────────────────────────────────────

def collect_dataset():
    """
    Walk through dataset/<language>/ folders and collect (filepath, label) pairs.
    Returns X (feature matrix) and y (label array).
    """
    features_list = []
    labels_list   = []
    file_counts   = {}

    for lang in LANGUAGES:
        lang_dir = os.path.join(DATASET_DIR, lang)
        os.makedirs(lang_dir, exist_ok=True)   # create folder if missing

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


def train(X, y):
    """Train a Random Forest and return the model, encoder, and metrics dict."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    if len(np.unique(y_enc)) < 2:
        raise ValueError("Need samples from at least 2 languages to train.")

    # If very few samples, skip stratified split
    test_size = 0.2 if len(X) >= 10 else 0.1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=42, stratify=y_enc
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "accuracy":       round(acc * 100, 2),
        "report":         report,
        "confusion_matrix": cm,
        "classes":        le.classes_.tolist(),
        "n_train":        len(X_train),
        "n_test":         len(X_test),
    }

    return clf, le, metrics


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Voice Language Identification - Training Script")
    print("=" * 60)

    # 1. Collect data
    print("\n[1/4] Scanning dataset folders ...")
    X, y, file_counts = collect_dataset()

    if len(X) == 0:
        print("\n[ERROR] No audio files found in the dataset folders.")
        print("        Place WAV files in dataset/<language>/ and re-run.\n")
        sys.exit(1)

    print(f"\n      Total samples collected: {len(X)}")

    # 2. Train
    print("\n[2/4] Training Random Forest classifier ...")
    try:
        clf, le, metrics = train(X, y)
    except ValueError as e:
        print(f"\n[ERROR] Training error: {e}\n")
        sys.exit(1)

    print(f"      Accuracy on test set: {metrics['accuracy']}%")

    # 3. Save model artefacts
    print("\n[3/4] Saving model ...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le,  ENCODER_PATH)

    # Save report for the web UI to read
    report_data = {
        **metrics,
        "file_counts": file_counts,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"      Model   : {MODEL_PATH}")
    print(f"      Encoder : {ENCODER_PATH}")
    print(f"      Report  : {REPORT_PATH}")

    # 4. Print classification report
    print("\n[4/4] Evaluation\n")
    header = f"{'Language':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}"
    print(header)
    print("-" * len(header))
    for lang in metrics["classes"]:
        r = metrics["report"].get(lang, {})
        print(
            f"{lang:<12} "
            f"{r.get('precision', 0):>10.2f} "
            f"{r.get('recall', 0):>8.2f} "
            f"{r.get('f1-score', 0):>8.2f} "
            f"{int(r.get('support', 0)):>9}"
        )

    print(f"\n  Overall Accuracy: {metrics['accuracy']}%")
    print("\n[DONE] Training completed successfully!\n")
    print("       Run the web app with:  python app.py\n")


if __name__ == "__main__":
    main()
