"""
predict.py
Prediction module for the Voice Language Identification System.

Usage (CLI):
    python predict.py path/to/audio.wav

Or import and call predict_language() from app.py.
"""

import os
import sys
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.audio_processing import load_audio, preprocess_audio, validate_audio
from utils.feature_extraction import extract_mfcc_features

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(BASE_DIR, "models")
MODEL_PATH   = os.path.join(MODELS_DIR, "language_model.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")


def load_model():
    """Load the trained model and label encoder. Raises FileNotFoundError if missing."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not trained yet. Please train the model first."
        )
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(
            "Label encoder not found. Please re-train the model."
        )
    clf = joblib.load(MODEL_PATH)
    le  = joblib.load(ENCODER_PATH)
    return clf, le


def predict_language(filepath):
    """
    Predict the language of the given WAV file.

    Returns a dict:
    {
        "language":   "Tamil",
        "confidence": 87.42,
        "all_probs":  {"English": 5.0, "Hindi": 3.0, "Tamil": 87.42, "Telugu": 4.58},
        "duration":   4.8,
        "sample_rate": 16000,
        "filename":   "sample.wav",
    }
    """
    # ── Load model ─────────────────────────────────────────────────────────────
    clf, le = load_model()

    # ── Load & preprocess audio ────────────────────────────────────────────────
    audio, sr = load_audio(filepath)
    audio     = preprocess_audio(audio, sr)
    validate_audio(audio, sr)

    duration = round(len(audio) / sr, 2)

    # ── Extract features ───────────────────────────────────────────────────────
    features = extract_mfcc_features(audio, sr).reshape(1, -1)

    # ── Predict ────────────────────────────────────────────────────────────────
    proba     = clf.predict_proba(features)[0]          # shape: (n_classes,)
    pred_idx  = int(np.argmax(proba))
    confidence = round(float(proba[pred_idx]) * 100, 2)

    language  = le.inverse_transform([pred_idx])[0]
    language  = language.capitalize()

    all_probs = {
        le.inverse_transform([i])[0].capitalize(): round(float(p) * 100, 2)
        for i, p in enumerate(proba)
    }

    return {
        "language":    language,
        "confidence":  confidence,
        "all_probs":   all_probs,
        "duration":    duration,
        "sample_rate": sr,
        "filename":    os.path.basename(filepath),
    }


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_audio.wav>")
        sys.exit(1)

    audio_path = sys.argv[1]

    try:
        result = predict_language(audio_path)
        print(f"\nFile           : {result['filename']}")
        print(f"Duration       : {result['duration']} seconds")
        print(f"Sampling Rate  : {result['sample_rate']} Hz")
        print(f"\nPredicted Language : {result['language']}")
        print(f"Confidence         : {result['confidence']}%")
        print("\nProbabilities per language:")
        for lang, prob in sorted(result["all_probs"].items(), key=lambda x: -x[1]):
            bar = "#" * int(prob / 5)
            print(f"  {lang:<10} {prob:>6.2f}%  {bar}")
    except FileNotFoundError as e:
        print(f"\n[!]  {e}")
    except ValueError as e:
        print(f"\n[!]  {e}")
    except Exception as e:
        print(f"\n[X] Unexpected error: {e}")
