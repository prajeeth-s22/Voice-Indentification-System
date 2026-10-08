"""
audio_processing.py
Handles audio loading, preprocessing, and validation with robust Vercel serverless fallbacks.
"""

import os
import numpy as np

# Constants
TARGET_SR = 16000
MIN_DURATION = 0.5  # seconds


def load_audio(filepath, target_sr=TARGET_SR):
    """
    Load an audio file and resample to target sample rate.
    Uses librosa if available, falling back to scipy.io.wavfile.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in [".wav", ".WAV"]:
        raise ValueError(f"Unsupported format '{ext}'. Please upload a WAV file.")

    # Try librosa first
    try:
        import librosa
        audio, sr = librosa.load(filepath, sr=target_sr, mono=True)
        return audio, sr
    except Exception:
        pass

    # Fallback to scipy.io.wavfile (zero C-library dependency)
    try:
        from scipy.io import wavfile
        from scipy import signal

        sr, data = wavfile.read(filepath)
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128.0) / 128.0
        else:
            data = data.astype(np.float32)

        if sr != target_sr:
            num_samples = int(len(data) * target_sr / sr)
            data = signal.resample(data, num_samples)
            sr = target_sr

        return data.astype(np.float32), sr
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file: {e}")


def preprocess_audio(audio, sr=TARGET_SR):
    """
    Preprocess audio:
      1. Normalize amplitude to [-1, 1]
      2. Trim leading/trailing silence
    Returns processed audio array.
    """
    try:
        import librosa
        audio, _ = librosa.effects.trim(audio, top_db=25)
    except Exception:
        # Simple energy-based silence trimming fallback
        energy = np.abs(audio)
        threshold = 0.01 * np.max(energy) if len(energy) > 0 else 0.01
        mask = energy > threshold
        if np.any(mask):
            start = np.argmax(mask)
            end = len(mask) - np.argmax(mask[::-1])
            audio = audio[start:end]

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
