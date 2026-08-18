# Voice Language Identification System

**Speech & Language Processing Mini Project**

---

## Description

This system identifies the spoken language from a short audio recording using
**MFCC (Mel-Frequency Cepstral Coefficients)** features and a
**Random Forest** classifier.

Supported Languages: **English · Hindi · Tamil · Telugu**

---

## Features

| Feature | Status |
|---|---|
| WAV file upload via web UI | ✅ |
| Audio preprocessing (mono, 16 kHz, normalize, trim) | ✅ |
| MFCC feature extraction (40 coefficients → 80-dim vector) | ✅ |
| Random Forest classification | ✅ |
| Language prediction with confidence score | ✅ |
| Per-language probability display | ✅ |
| Accuracy, Precision, Recall, F1-Score evaluation | ✅ |
| Confusion matrix | ✅ |
| Browser-based training trigger | ✅ |
| Error handling (no model, bad format, short audio) | ✅ |

---

## Project Structure

```
voice-language-identification/
│
├── app.py                  # Flask web application
├── train_model.py          # Training script
├── predict.py              # Prediction module (also CLI)
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── english/            ← place English WAV files here
│   ├── hindi/              ← place Hindi   WAV files here
│   ├── tamil/              ← place Tamil   WAV files here
│   └── telugu/             ← place Telugu  WAV files here
│
├── models/
│   ├── language_model.pkl  (generated after training)
│   └── label_encoder.pkl   (generated after training)
│
├── utils/
│   ├── audio_processing.py
│   └── feature_extraction.py
│
└── static/
    └── styles.css
```

---

## Installation

```bash
pip install -r requirements.txt
```

> Requires Python 3.8+

---

## Dataset Setup

1. Collect short WAV recordings (3–10 seconds) for each language.
2. Place them in the corresponding folder:

```
dataset/english/  →  en_001.wav, en_002.wav, …
dataset/hindi/    →  hi_001.wav, hi_002.wav, …
dataset/tamil/    →  ta_001.wav, ta_002.wav, …
dataset/telugu/   →  te_001.wav, te_002.wav, …
```

> **Minimum:** 2 files per language to train.  
> **Recommended:** 20–50 files per language for good accuracy.

Useful free audio sources:
- Mozilla Common Voice: https://commonvoice.mozilla.org
- OpenSLR: https://openslr.org

---

## Run Training (Command Line)

```bash
python train_model.py
```

Sample output:
```
[1/4] Scanning dataset folders …
  ✔  'english': 15 file(s) found.
  ✔  'hindi':   12 file(s) found.
  ✔  'tamil':   14 file(s) found.
  ✔  'telugu':  13 file(s) found.

[2/4] Training Random Forest classifier …
      Accuracy on test set: 86.67%

[3/4] Saving model …

[4/4] Evaluation
Language     Precision   Recall       F1  Support
-------------------------------------------------------
english           0.89     0.89     0.89        9
hindi             0.83     0.83     0.83        6
tamil             0.88     0.88     0.88        8
telugu            0.86     0.86     0.86        8

✅  Training completed successfully!
```

---

## Run Application

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

You can also trigger training directly from the **Train Model** tab in the UI.

---

## CLI Prediction (Optional)

```bash
python predict.py path/to/audio.wav
```

Output:
```
File           : sample.wav
Duration       : 4.8 seconds
Sampling Rate  : 16000 Hz

Predicted Language : Tamil
Confidence         : 87.42%

Probabilities per language:
  Tamil      87.42%  █████████████████
  English     5.21%  █
  Hindi       3.98%
  Telugu      3.39%
```

---

## How It Works

```
User Audio (WAV)
       │
       ▼
Audio Preprocessing
 • Convert to mono
 • Resample to 16,000 Hz
 • Normalize amplitude
 • Trim silence
       │
       ▼
MFCC Feature Extraction
 • 40 MFCC coefficients
 • Compute mean (40 values)
 • Compute std deviation (40 values)
 • Concatenate → 80-dimensional feature vector
       │
       ▼
Random Forest Classifier
 (100 trees, trained on labelled audio)
       │
       ▼
Language Prediction + Confidence Score
```

**Why MFCCs?** MFCCs capture the spectral envelope of speech in a compact
representation that is highly discriminative for language identification.
Different languages have characteristic phoneme sets and prosodic patterns
that are reflected in the MFCC distribution.

---

## Limitations (Phase 1)

- Accuracy depends heavily on dataset size and quality
- Background noise can reduce accuracy
- Very short clips (< 1 second) may be unreliable
- Only 4 languages supported

---

## Future Work (Phase 2)

- CNN / LSTM deep learning models
- Wav2Vec2 / HuBERT pre-trained embeddings
- Real-time microphone identification
- More languages
- Noise robustness
- Mobile / cloud deployment

---

*Voice Language Identification System — Speech & Language Processing Mini Project*
