"""
generate_demo_audio.py
──────────────────────
Generates synthetic WAV samples for each language folder so that you can
test training and prediction immediately, without any real audio data.

Each "sample" is a short clip of band-limited noise with a unique
frequency profile per language (simulating different phoneme distributions).
These samples are purely for demonstration/smoke-testing purposes.

Usage:
    python generate_demo_audio.py

This will create 15 WAV files in each language folder (60 total).
"""

import os
import sys
import numpy as np

try:
    import soundfile as sf
except ImportError:
    print("soundfile not installed. Run: pip install soundfile")
    sys.exit(1)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
SR          = 16000
DURATION    = 4      # seconds per clip
N_PER_LANG  = 15     # files to generate per language
SEED        = 42

# Each language gets a unique "noise fingerprint":
# a set of dominant frequency bands (in Hz) that shift the spectral centroid.
LANG_PROFILES = {
    "english": {"center_freqs": [400,  1200, 3000], "noise_color": 1.0},
    "hindi":   {"center_freqs": [300,  900,  2500], "noise_color": 0.8},
    "tamil":   {"center_freqs": [250,  700,  2000], "noise_color": 0.6},
    "telugu":  {"center_freqs": [350,  1000, 2800], "noise_color": 0.9},
}

rng = np.random.default_rng(SEED)


def band_limited_noise(sr, duration, center_freqs, noise_color=1.0):
    """
    Generate a clip by mixing narrow-band sinusoids at center_freqs with
    coloured noise. noise_color: 1.0 = white, <1.0 = more pink/red.
    """
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)

    signal = np.zeros(n)

    # Add sinusoidal components (voiced-like)
    for fc in center_freqs:
        amp   = rng.uniform(0.2, 0.6)
        phase = rng.uniform(0, 2 * np.pi)
        signal += amp * np.sin(2 * np.pi * fc * t + phase)

    # Add coloured noise (unvoiced-like)
    white = rng.standard_normal(n)
    freqs  = np.fft.rfftfreq(n, d=1.0 / sr)
    fft_w  = np.fft.rfft(white)
    # Apply 1/f^alpha shaping
    alpha = 1 - noise_color
    with np.errstate(divide="ignore", invalid="ignore"):
        power = np.where(freqs == 0, 0, freqs ** (-alpha))
    fft_c = fft_w * power
    coloured = np.fft.irfft(fft_c, n=n)

    signal += 0.3 * coloured

    # Normalize
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal /= max_val

    # Small random amplitude variation (simulate prosody)
    envelope = 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * rng.uniform(0.5, 2.0) * t))
    signal  *= envelope

    return signal.astype(np.float32)


def main():
    print("Generating demo audio samples …\n")

    total = 0
    for lang, profile in LANG_PROFILES.items():
        lang_dir = os.path.join(DATASET_DIR, lang)
        os.makedirs(lang_dir, exist_ok=True)

        for i in range(1, N_PER_LANG + 1):
            filepath = os.path.join(lang_dir, f"{lang}_{i:03d}.wav")
            audio    = band_limited_noise(SR, DURATION, **profile)
            sf.write(filepath, audio, SR)
            total   += 1

        print(f"  [OK] {lang:<10} - {N_PER_LANG} files written to dataset/{lang}/")

    print(f"\n  Total: {total} demo WAV files created.")
    print("\n  You can now run:")
    print("    python train_model.py   (train the model)")
    print("    python app.py           (start the web app)\n")


if __name__ == "__main__":
    main()
