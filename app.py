"""
app.py
──────
Flask web application for the Voice Language Identification System.
Features:
- Multi-Model Classification Architecture:
    1. Local Classical ML Suite (MFCC features + Model Selection Layer):
       - Random Forest
       - Support Vector Machine (SVM)
       - K-Nearest Neighbors (KNN)
       - Logistic Regression
       - Decision Tree
       - Gradient Boosting
       - Auto – Best Performing Model
    2. Google Gemini AI (Multimodal Audio analysis for real-world speech)
- Model Comparison Dashboard (Accuracy, Precision, Recall, F1, Confusion Matrix)
- Dynamic Best-Performing Model Indicator
- Live Audio Recording directly in browser
- API Key management UI
- Local Training & Evaluation dashboards

Run with:
    python app.py
Then open:  http://127.0.0.1:5000
"""

import os
import sys
import json
import tempfile
import subprocess
import threading

from flask import (
    Flask, render_template, request, jsonify, send_from_directory
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.gemini_service import (
    get_gemini_api_key,
    set_gemini_api_key_env,
)
from utils.model_registry import (
    MODEL_DEFINITIONS,
    get_model_choices,
    clear_model_cache,
)
from predict import predict_language

# ── App setup ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
REPORT_PATH = os.path.join(MODELS_DIR, "training_report.json")
LANGUAGES   = ["english", "hindi", "tamil", "telugu"]

ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "flac", "m4a", "aac", "webm"}

# Create required dirs safely
try:
    for lang in LANGUAGES:
        os.makedirs(os.path.join(DATASET_DIR, lang), exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
except Exception:
    pass

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # 50 MB upload limit

# Make enumerate() available in Jinja2 templates
app.jinja_env.globals.update(enumerate=enumerate)

# ── Training state (thread-safe) ───────────────────────────────────────────────
_training_state = {"status": "idle", "message": "", "result": None}
_training_lock  = threading.Lock()


# ── Helpers ────────────────────────────────────────────────────────────────────

def allowed_file(filename):
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def get_dataset_counts():
    counts = {}
    for lang in LANGUAGES:
        lang_dir = os.path.join(DATASET_DIR, lang)
        wav_files = [
            f for f in os.listdir(lang_dir) if f.lower().endswith(".wav")
        ] if os.path.isdir(lang_dir) else []
        counts[lang] = len(wav_files)
    return counts


def model_exists():
    encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
    rf_path      = os.path.join(MODELS_DIR, "random_forest.pkl")
    legacy_path  = os.path.join(MODELS_DIR, "language_model.pkl")
    has_model = os.path.exists(rf_path) or os.path.exists(legacy_path)
    return has_model and os.path.exists(encoder_path)


def load_training_report():
    if os.path.exists(REPORT_PATH):
        try:
            with open(REPORT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def mask_key(k: str) -> str:
    if not k or len(k) < 8:
        return "Not configured"
    return f"{k[:4]}...{k[-4:]}"


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    active_key = get_gemini_api_key()
    return render_template(
        "index.html",
        dataset_counts=get_dataset_counts(),
        model_trained=model_exists(),
        training_report=load_training_report(),
        has_gemini_key=bool(active_key),
        masked_key=mask_key(active_key),
        model_choices=get_model_choices(),
        model_definitions=MODEL_DEFINITIONS,
    )


@app.route("/models", methods=["GET"])
def list_models():
    """Return available model choices and training report overview."""
    report = load_training_report()
    return jsonify({
        "choices": [
            {"id": cid, "label": cname} for cid, cname in get_model_choices()
        ],
        "definitions": {
            k: {
                "id": v["id"],
                "name": v["name"],
                "description": v["description"],
                "is_trained": os.path.exists(os.path.join(MODELS_DIR, v["filename"]))
            }
            for k, v in MODEL_DEFINITIONS.items()
        },
        "best_model": report.get("best_model") if report else None,
        "best_model_name": report.get("best_model_name") if report else None,
    })


@app.route("/api_key_status", methods=["GET"])
def api_key_status():
    key = get_gemini_api_key()
    return jsonify({
        "configured": bool(key),
        "masked_key": mask_key(key) if key else "",
    })


@app.route("/save_api_key", methods=["POST"])
def save_api_key():
    data = request.get_json() or {}
    key = data.get("api_key", "").strip()
    if not key:
        return jsonify({"error": "API key cannot be empty."}), 400

    set_gemini_api_key_env(key)
    return jsonify({
        "success": True,
        "message": "Gemini API key saved successfully.",
        "masked_key": mask_key(key),
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Accept audio file or recording, run prediction with chosen engine and model."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded. Please select or record an audio file."}), 400

    f = request.files["audio"]
    if f.filename == "":
        return jsonify({"error": "No audio file selected."}), 400

    if not allowed_file(f.filename):
        return jsonify({"error": f"Unsupported format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    engine = request.form.get("engine", "gemini")
    model_choice = request.form.get("model", request.form.get("model_key", "random_forest"))
    user_api_key = request.form.get("api_key", "").strip()

    # Determine file extension
    ext = os.path.splitext(f.filename)[1].lower() or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    try:
        f.save(tmp.name)
        tmp.close()

        result = predict_language(
            tmp.name,
            engine=engine,
            api_key=user_api_key if user_api_key else None,
            model=model_choice,
        )
        return jsonify({"success": True, "result": result})

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


@app.route("/train", methods=["POST"])
def train():
    """Trigger training in a background thread."""
    global _training_state

    with _training_lock:
        if _training_state["status"] == "running":
            return jsonify({"error": "Training is already in progress."}), 409
        _training_state = {"status": "running", "message": "Training started…", "result": None}

    def _run_training():
        global _training_state
        script = os.path.join(BASE_DIR, "train_model.py")
        try:
            proc = subprocess.run(
                [sys.executable, script],
                capture_output=True, text=True, cwd=BASE_DIR
            )
            clear_model_cache()
            if proc.returncode == 0:
                report = load_training_report()
                with _training_lock:
                    _training_state = {
                        "status": "done",
                        "message": "Training completed successfully.",
                        "result": report,
                    }
            else:
                err = (proc.stderr or proc.stdout or "Unknown error").strip()
                lines = [l for l in err.splitlines() if l.strip()]
                msg = lines[-1] if lines else err
                with _training_lock:
                    _training_state = {
                        "status": "error",
                        "message": msg,
                        "result": None,
                    }
        except Exception as e:
            with _training_lock:
                _training_state = {
                    "status": "error",
                    "message": str(e),
                    "result": None,
                }

    thread = threading.Thread(target=_run_training, daemon=True)
    thread.start()

    return jsonify({"message": "Training started."})


@app.route("/train_status")
def train_status():
    """Poll training progress."""
    with _training_lock:
        return jsonify(dict(_training_state))


@app.route("/dataset_counts")
def dataset_counts():
    return jsonify(get_dataset_counts())


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("\n" + "=" * 65)
    print("  Voice Language Identification System (Multi-Model ML + Gemini AI)")
    print("=" * 65)
    print("  Running on: http://127.0.0.1:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
