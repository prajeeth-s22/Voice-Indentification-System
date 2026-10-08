"""
gemini_service.py
─────────────────
Provides Voice Language Identification using Google Gemini Multimodal Audio AI.
Supports Gemini 2.0 Flash / Gemini 1.5 Flash models.

Features:
- Identifies the spoken language (English, Hindi, Tamil, Telugu, or any global/regional language)
- Generates precise transcription in native script
- Provides English translation
- Calculates realistic confidence score and comparative language probabilities
- Analyzes phonetic markers, dialect, and prosody
"""

import os
import json
import base64
import mimetypes
import requests
from pathlib import Path

# Supported Gemini models in priority order
DEFAULT_MODELS = [
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
]

# Supported candidate languages for standard comparison
STANDARD_LANGUAGES = ["English", "Hindi", "Tamil", "Telugu"]


def fetch_available_gemini_models(key: str) -> list:
    """
    Query Gemini ModelService (ListModels) to retrieve active models that support generateContent.
    """
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
        resp = requests.get(url, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("models", [])
            valid_models = []
            for m in models:
                methods = m.get("supportedGenerationMethods", [])
                name = m.get("name", "").replace("models/", "")
                if "generateContent" in methods:
                    valid_models.append(name)
            return valid_models
    except Exception:
        pass
    return []


def get_gemini_api_key(explicit_key: str = None) -> str:
    """
    Retrieve Gemini API key from explicit param, environment, or .env file.
    """
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()

    # Check environment variable
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key and key.strip():
        return key.strip()

    # Check local .env file
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("#") or not line:
                        continue
                    if line.startswith("GEMINI_API_KEY=") or line.startswith("GOOGLE_API_KEY="):
                        k = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if k:
                            return k
        except Exception:
            pass

    return ""


def set_gemini_api_key_env(api_key: str) -> None:
    """
    Persist API key to .env file and environment variable.
    """
    api_key = api_key.strip()
    os.environ["GEMINI_API_KEY"] = api_key
    env_file = Path(__file__).resolve().parent.parent / ".env"
    try:
        lines = []
        found = False
        if env_file.exists():
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("GEMINI_API_KEY=") or line.startswith("GOOGLE_API_KEY="):
                        lines.append(f"GEMINI_API_KEY={api_key}\n")
                        found = True
                    else:
                        lines.append(line)
        if not found:
            lines.append(f"GEMINI_API_KEY={api_key}\n")

        with open(env_file, "w", encoding="utf-8") as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Warning: Could not save API key to .env: {e}")


def get_audio_mime_type(filepath: str) -> str:
    """Determine MIME type for the audio file."""
    ext = os.path.splitext(filepath)[1].lower()
    mapping = {
        ".wav": "audio/wav",
        ".mp3": "audio/mp3",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".m4a": "audio/m4a",
        ".aac": "audio/aac",
        ".webm": "audio/webm",
    }
    return mapping.get(ext, mimetypes.guess_type(filepath)[0] or "audio/wav")


def predict_language_with_gemini(
    filepath: str,
    api_key: str = None,
    preferred_model: str = None
) -> dict:
    """
    Identifies language from voice audio file using Google Gemini Multimodal Audio API.
    
    Returns structured result:
    {
        "language": "Hindi",
        "confidence": 97.5,
        "transcript": "नमस्ते, आप कैसे हैं?",
        "english_translation": "Hello, how are you?",
        "dialect_or_accent": "Standard Hindi with polite formal register",
        "reasoning": "Identified characteristic Devanagari phonology and greeting 'Namaste'.",
        "all_probs": {"Hindi": 97.5, "English": 1.2, "Tamil": 0.8, "Telugu": 0.5},
        "duration": 3.4,
        "sample_rate": 16000,
        "filename": "sample.wav",
        "model_used": "gemini-2.0-flash",
        "engine": "gemini"
    }
    """
    key = get_gemini_api_key(api_key)
    if not key:
        raise ValueError(
            "Gemini API key is required. Please set GEMINI_API_KEY in your environment or enter it in the web interface."
        )

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    # Read and encode audio file
    with open(filepath, "rb") as f:
        audio_bytes = f.read()

    b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
    mime_type = get_audio_mime_type(filepath)

    # Compute audio duration if possible
    duration = 0.0
    sample_rate = 16000
    try:
        import soundfile as sf
        info = sf.info(filepath)
        duration = round(info.duration, 2)
        sample_rate = info.samplerate
    except Exception:
        try:
            import librosa
            y, sr = librosa.load(filepath, sr=None)
            duration = round(len(y) / sr, 2)
            sample_rate = sr
        except Exception:
            duration = round(len(audio_bytes) / 32000, 2)

    prompt = (
        "You are an expert computational linguist and speech recognition AI. "
        "Analyze the provided audio recording carefully.\n\n"
        "Tasks:\n"
        "1. Detect the exact language spoken (e.g. English, Hindi, Tamil, Telugu, or any other specific language).\n"
        "2. Accurately transcribe what was spoken into the native script of the spoken language.\n"
        "3. Provide the English translation of what was spoken.\n"
        "4. Estimate confidence score (0 to 100).\n"
        "5. Provide estimated probability distribution percentages across major candidate languages "
        "(including English, Hindi, Tamil, Telugu, and any other relevant language detected) so the probabilities sum to ~100%.\n"
        "6. Provide a concise explanation of phonetic cues, phonemes, or vocabulary that helped identify the language.\n"
        "7. Note any accent or dialect characteristics.\n\n"
        "Output ONLY a raw, valid JSON object without any Markdown fences or backticks, matching this exact schema:\n"
        "{\n"
        '  "language": "string (Capitalized language name, e.g. Hindi)",\n'
        '  "language_code": "string (e.g. hi-IN, en-US, ta-IN, te-IN)",\n'
        '  "confidence": 98.0,\n'
        '  "transcript": "string (transcript in native script)",\n'
        '  "english_translation": "string (English translation)",\n'
        '  "dialect_or_accent": "string (dialect or accent notes)",\n'
        '  "reasoning": "string (short linguistic/phonetic reasoning)",\n'
        '  "all_probs": {\n'
        '    "English": 2.0,\n'
        '    "Hindi": 95.0,\n'
        '    "Tamil": 1.5,\n'
        '    "Telugu": 1.5\n'
        "  }\n"
        "}"
    )

    # Dynamically discover active models from API + priority defaults
    online_available = fetch_available_gemini_models(key)

    candidate_pool = []
    if preferred_model:
        candidate_pool.append(preferred_model)

    # Add standard working defaults first
    for m in DEFAULT_MODELS:
        if m not in candidate_pool:
            candidate_pool.append(m)

    # Add any extra valid online models discovered from API key scope
    for m in online_available:
        if m not in candidate_pool:
            candidate_pool.append(m)

    models_to_try = candidate_pool
    last_err = None

    for model_name in models_to_try:
        if not model_name:
            continue
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": b64_audio,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "response_mime_type": "application/json",
            },
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=40)
            if resp.status_code == 200:
                data = resp.json()
                # Parse candidate text
                candidates = data.get("candidates", [])
                if not candidates:
                    continue
                content_parts = candidates[0].get("content", {}).get("parts", [])
                if not content_parts:
                    continue
                raw_text = content_parts[0].get("text", "").strip()

                # Clean markdown backticks if any
                if raw_text.startswith("```"):
                    raw_text = raw_text.strip("`")
                    if raw_text.startswith("json"):
                        raw_text = raw_text[4:].strip()

                parsed = json.loads(raw_text)

                # Ensure required fields and sane defaults
                lang = parsed.get("language", "Unknown").capitalize()
                confidence = float(parsed.get("confidence", 95.0))
                transcript = parsed.get("transcript", "")
                translation = parsed.get("english_translation", "")
                dialect = parsed.get("dialect_or_accent", "Standard")
                reasoning = parsed.get("reasoning", "")
                probs = parsed.get("all_probs", {})

                # Ensure standard languages are present in probs
                if not probs:
                    probs = {lang: confidence}
                for std_lang in STANDARD_LANGUAGES:
                    if std_lang not in probs:
                        rem = max(0.5, round((100.0 - confidence) / len(STANDARD_LANGUAGES), 1))
                        probs[std_lang] = rem
                # Normalize language casing in probs
                formatted_probs = {k.capitalize(): float(v) for k, v in probs.items()}

                return {
                    "language": lang,
                    "language_code": parsed.get("language_code", ""),
                    "confidence": round(confidence, 2),
                    "transcript": transcript,
                    "english_translation": translation,
                    "dialect_or_accent": dialect,
                    "reasoning": reasoning,
                    "all_probs": formatted_probs,
                    "duration": duration,
                    "sample_rate": sample_rate,
                    "filename": os.path.basename(filepath),
                    "model_used": model_name,
                    "engine": "gemini",
                }
            else:
                error_msg = resp.text
                try:
                    err_json = resp.json()
                    error_msg = err_json.get("error", {}).get("message", error_msg)
                except Exception:
                    pass
                last_err = f"{resp.status_code} - {error_msg}"
        except Exception as e:
            last_err = str(e)

    raise RuntimeError(f"Gemini API request failed on all attempted models. Last error: {last_err}")
