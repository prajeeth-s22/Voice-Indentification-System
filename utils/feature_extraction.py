"""
feature_extraction.py
Extracts MFCC-based features from preprocessed audio.
"""

import numpy as np
import librosa

N_MFCC = 40


def extract_mfcc_features(audio, sr=16000, n_mfcc=N_MFCC):
    """
    Extract MFCC features from an audio array.

    Returns an 80-dimensional feature vector:
      - 40 MFCC mean values
      - 40 MFCC std-deviation values
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    mfcc_mean = np.mean(mfcc, axis=1)   # shape: (n_mfcc,)
    mfcc_std  = np.std(mfcc,  axis=1)   # shape: (n_mfcc,)

    features = np.concatenate([mfcc_mean, mfcc_std])  # shape: (2*n_mfcc,)
    return features


def extract_features_from_file(filepath, preprocess_fn=None, sr=16000):
    """
    Convenience wrapper: load → preprocess → extract features.

    preprocess_fn: callable(audio, sr) -> audio   (optional)
    Returns feature vector (numpy array).
    """
    import librosa as _librosa

    audio, loaded_sr = _librosa.load(filepath, sr=sr, mono=True)

    if preprocess_fn is not None:
        audio = preprocess_fn(audio, loaded_sr)

    return extract_mfcc_features(audio, sr=loaded_sr)
