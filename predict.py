"""
predict.py
──────────
Prediction module for the Voice Language Identification System.
Supports:
  1. Multiple Local Machine Learning Models:
     - Random Forest
     - Support Vector Machine (SVM)
     - K-Nearest Neighbors (KNN)
     - Logistic Regression
     - Decision Tree
     - Gradient Boosting
     - Auto - Best Performing Model
  2. Google Gemini AI (Multimodal Audio analysis for real-world speech)

Usage (CLI):
    python predict.py path/to/audio.wav
    python predict.py path/to/audio.wav --model svm
    python predict.py path/to/audio.wav --model auto
    python predict.py path/to/audio.wav --engine gemini --api-key YOUR_KEY
"""

import os
import sys
import json
import argparse
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.audio_processing import load_audio, preprocess_audio, validate_audio
from utils.feature_extraction import extract_mfcc_features
from utils.model_registry import (
    MODEL_DEFINITIONS,
    load_classifier,
    normalize_model_key,
)
from utils.gemini_service import (
    predict_language_with_gemini,
    get_gemini_api_key,
    set_gemini_api_key_env,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(BASE_DIR, "models")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
REPORT_PATH  = os.path.join(MODELS_DIR, "training_report.json")

# In-memory cached label encoder
_LABEL_ENCODER = None


def load_label_encoder():
    """Load the LabelEncoder saved during training."""
    global _LABEL_ENCODER
    if _LABEL_ENCODER is not None:
        return _LABEL_ENCODER

    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(
            "Label encoder not found in models/. Please train the models first using 'python train_model.py'."
        )
    _LABEL_ENCODER = joblib.load(ENCODER_PATH)
    return _LABEL_ENCODER


def get_best_model_key():
    """Read the best-performing model key dynamically from training_report.json."""
    if os.path.exists(REPORT_PATH):
        try:
            with open(REPORT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                best_id = data.get("best_model")
                if best_id and best_id in MODEL_DEFINITIONS:
                    return best_id
        except Exception:
            pass
    return "random_forest"


def resolve_model_key(requested_key: str):
    """
    Resolve 'auto' or alias to canonical model ID.
    Returns (resolved_key, is_auto, display_name).
    """
    normalized = normalize_model_key(requested_key)
    is_auto = (normalized == "auto")

    if is_auto:
        actual_key = get_best_model_key()
    else:
        actual_key = normalized if normalized in MODEL_DEFINITIONS else "random_forest"

    model_info = MODEL_DEFINITIONS.get(actual_key, MODEL_DEFINITIONS["random_forest"])
    display_name = model_info["name"]
    if is_auto:
        display_name = f"{display_name} (Auto – Best Model)"

    return actual_key, is_auto, display_name


def predict_language_local(filepath, model_key="random_forest"):
    """
    Predict language using the selected local classification model.

    Args:
        filepath (str): Path to audio file.
        model_key (str): Chosen model identifier (e.g. 'svm', 'random_forest', 'auto').

    Returns:
        dict: Standardized prediction result.
    """
    resolved_key, is_auto, display_name = resolve_model_key(model_key)

    # 1. Load classifier and label encoder
    clf = load_classifier(resolved_key)
    le  = load_label_encoder()

    # 2. Load and preprocess audio
    audio, sr = load_audio(filepath)
    audio     = preprocess_audio(audio, sr)
    validate_audio(audio, sr)

    duration = round(len(audio) / sr, 2)

    # 3. Extract 80-dim MFCC feature vector
    features = extract_mfcc_features(audio, sr).reshape(1, -1)

    # 4. Predict language & compute confidence
    confidence = None
    all_probs = {}

    if hasattr(clf, "predict_proba"):
        try:
            proba = clf.predict_proba(features)[0]
            pred_idx = int(np.argmax(proba))
            confidence = round(float(proba[pred_idx]) * 100, 2)
            language = le.inverse_transform([pred_idx])[0].capitalize()

            all_probs = {
                le.inverse_transform([i])[0].capitalize(): round(float(p) * 100, 2)
                for i, p in enumerate(proba)
            }
        except Exception:
            # Fallback if probability fails
            pred_idx = clf.predict(features)[0]
            language = le.inverse_transform([pred_idx])[0].capitalize()
    else:
        pred_idx = clf.predict(features)[0]
        language = le.inverse_transform([pred_idx])[0].capitalize()

    return {
        "language":    language,
        "confidence":  confidence if confidence is not None else "N/A",
        "all_probs":   all_probs,
        "duration":    duration,
        "sample_rate": sr,
        "filename":    os.path.basename(filepath),
        "engine":      "local",
        "model_key":   resolved_key,
        "model_used":  display_name,
        "is_auto":     is_auto,
    }


def predict_language(filepath, engine="auto", api_key=None, preferred_model=None, model="random_forest"):
    """
    High-level prediction dispatch.

    engine:
      - 'auto': Use Gemini AI if API key is configured; fallback to local classifier.
      - 'gemini': Use Google Gemini Multimodal Audio model.
      - 'local': Use selected local classification model.

    model / model_key:
      - 'random_forest', 'svm', 'knn', 'logistic_regression', 'decision_tree', 'gradient_boosting', 'auto'
    """
    gemini_key = get_gemini_api_key(api_key)

    if engine == "gemini":
        if not gemini_key:
            raise ValueError(
                "Gemini API key is not configured. Set GEMINI_API_KEY environment variable "
                "or provide --api-key in the CLI/web interface."
            )
        return predict_language_with_gemini(filepath, api_key=gemini_key, preferred_model=preferred_model)

    if engine == "local":
        return predict_language_local(filepath, model_key=model)

    # Auto engine dispatch:
    if gemini_key:
        try:
            return predict_language_with_gemini(filepath, api_key=gemini_key, preferred_model=preferred_model)
        except Exception as e:
            print(f"[!] Gemini inference encountered error: {e}. Falling back to local classifier.")
            return predict_language_local(filepath, model_key=model)
    else:
        return predict_language_local(filepath, model_key=model)


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Identify spoken language from audio using Multiple Classifiers or Gemini AI."
    )
    parser.add_argument("audio_path", help="Path to audio file (.wav, .mp3, etc.)")
    parser.add_argument(
        "--engine",
        choices=["auto", "gemini", "local"],
        default="local",
        help="Recognition engine: 'local' (offline ML classifiers) or 'gemini' (cloud AI)",
    )
    parser.add_argument(
        "--model",
        choices=[
            "random_forest",
            "svm",
            "knn",
            "logistic_regression",
            "decision_tree",
            "gradient_boosting",
            "auto",
        ],
        default="random_forest",
        help="Classification model to use for local recognition (default: random_forest)",
    )
    parser.add_argument("--api-key", help="Gemini API Key (optional, can also use GEMINI_API_KEY env var)")
    parser.add_argument("--save-key", action="store_true", help="Save the provided API key to .env file for future use")

    args = parser.parse_args()

    if args.api_key and args.save_key:
        set_gemini_api_key_env(args.api_key)
        print("✓ Gemini API key saved to .env")

    try:
        result = predict_language(
            args.audio_path,
            engine=args.engine,
            api_key=args.api_key,
            model=args.model,
        )

        engine_name = "Google Gemini AI" if result.get("engine") == "gemini" else f"Local ML [{result.get('model_used')}]"
        print("\n" + "=" * 65)
        print(f"  Voice Language Identification Result")
        print(f"  Engine: {engine_name}")
        print("=" * 65)
        print(f"File           : {result.get('filename')}")
        print(f"Duration       : {result.get('duration')} seconds")
        print(f"Sampling Rate  : {result.get('sample_rate')} Hz")
        print(f"\nPredicted Language : {result.get('language')}")
        print(f"Confidence         : {result.get('confidence')}%")

        if result.get("transcript"):
            print(f"\nTranscript (Native): {result.get('transcript')}")
        if result.get("english_translation"):
            print(f"English Translation: {result.get('english_translation')}")
        if result.get("dialect_or_accent"):
            print(f"Accent / Dialect   : {result.get('dialect_or_accent')}")
        if result.get("reasoning"):
            print(f"Phonetic Reasoning : {result.get('reasoning')}")

        if result.get("all_probs"):
            print("\nProbabilities per language:")
            for lang, prob in sorted(result.get("all_probs", {}).items(), key=lambda x: -x[1]):
                bar = "#" * int(prob / 5)
                print(f"  {lang:<12} {prob:>6.2f}%  {bar}")
        print("=" * 65 + "\n")

    except FileNotFoundError as e:
        print(f"\n[!] File Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n[!] Configuration Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[X] Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
