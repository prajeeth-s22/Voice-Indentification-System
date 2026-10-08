"""
feature_extraction.py
Extracts MFCC-based features from preprocessed audio with fallback capability.
"""

import numpy as np

N_MFCC = 40


def extract_mfcc_features(audio, sr=16000, n_mfcc=N_MFCC):
    """
    Extract MFCC features from an audio array.

    Returns an 80-dimensional feature vector:
      - 40 MFCC mean values
      - 40 MFCC std-deviation values
    """
    try:
        import librosa
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    except Exception:
        # Fallback FFT-based Mel-like frequency spectrum representation
        fft_vals = np.abs(np.fft.rfft(audio, n=512))
        if len(fft_vals) < n_mfcc * 4:
            fft_vals = np.pad(fft_vals, (0, n_mfcc * 4 - len(fft_vals)))
        mfcc = fft_vals[: n_mfcc * 4].reshape(n_mfcc, 4)

    mfcc_mean = np.mean(mfcc, axis=1)   # shape: (n_mfcc,)
    mfcc_std  = np.std(mfcc,  axis=1)   # shape: (n_mfcc,)

    features = np.concatenate([mfcc_mean, mfcc_std])  # shape: (2*n_mfcc,)
    return features


def extract_features_from_file(filepath, preprocess_fn=None, sr=16000):
    """
    Convenience wrapper: load → preprocess → extract features.
    """
    from utils.audio_processing import load_audio

    audio, loaded_sr = load_audio(filepath, target_sr=sr)

    if preprocess_fn is not None:
        audio = preprocess_fn(audio, loaded_sr)

    return extract_mfcc_features(audio, sr=loaded_sr)
