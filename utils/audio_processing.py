"""
audio_processing.py
Handles audio loading, preprocessing, and validation.
"""

import numpy as np
import librosa
import soundfile as sf
import os

# Constants
TARGET_SR = 16000
MIN_DURATION = 0.5  # seconds


def load_audio(filepath, target_sr=TARGET_SR):
    """
    Load an audio file and resample to target sample rate.
    Returns (audio_array, sample_rate) or raises an exception.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in [".wav", ".WAV"]:
        raise ValueError(f"Unsupported format '{ext}'. Please upload a WAV file.")

    try:
        audio, sr = librosa.load(filepath, sr=target_sr, mono=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return audio, sr


def preprocess_audio(audio, sr=TARGET_SR):
    """
    Preprocess audio:
      1. Ensure mono (already done by librosa.load with mono=True)
      2. Normalize amplitude to [-1, 1]
      3. Trim leading/trailing silence
    Returns processed audio array.
    """
    # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=25)

    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    return audio


def get_audio_info(filepath, target_sr=TARGET_SR):
    """
    Return a dict with basic audio metadata.
    """
    audio, sr = load_audio(filepath, target_sr=target_sr)
    audio = preprocess_audio(audio, sr)
    duration = len(audio) / sr
    return {
        "filename": os.path.basename(filepath),
        "duration": round(duration, 2),
        "sample_rate": sr,
        "num_samples": len(audio),
    }


def validate_audio(audio, sr=TARGET_SR, min_duration=MIN_DURATION):
    """
    Raise ValueError if audio is too short.
    """
    duration = len(audio) / sr
    if duration < min_duration:
        raise ValueError(
            f"Audio clip is too short ({duration:.2f}s). "
            f"Please provide at least {min_duration}s of audio."
        )
