"""
app.py
Flask web application for the Voice Language Identification System.

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

from utils.audio_processing import load_audio, preprocess_audio, validate_audio
from utils.feature_extraction import extract_mfcc_features

# ── App setup ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
REPORT_PATH = os.path.join(MODELS_DIR, "training_report.json")
LANGUAGES   = ["english", "hindi", "tamil", "telugu"]

# Create required dirs
for lang in LANGUAGES:
    os.makedirs(os.path.join(DATASET_DIR, lang), exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # 50 MB upload limit

# Make enumerate() available in Jinja2 templates
app.jinja_env.globals.update(enumerate=enumerate)

# ── Training state (thread-safe) ───────────────────────────────────────────────
_training_state = {"status": "idle", "message": "", "result": None}
_training_lock  = threading.Lock()


# ── Helpers ────────────────────────────────────────────────────────────────────

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "wav"


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
    model_path   = os.path.join(MODELS_DIR, "language_model.pkl")
    encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
    return os.path.exists(model_path) and os.path.exists(encoder_path)


def load_training_report():
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH) as f:
            return json.load(f)
    return None


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        dataset_counts=get_dataset_counts(),
        model_trained=model_exists(),
        training_report=load_training_report(),
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Accept a WAV file, run prediction, return JSON."""
    if "audio" not in request.files:
        return jsonify({"error": "No file uploaded. Please select a WAV file."}), 400

    f = request.files["audio"]

    if f.filename == "":
        return jsonify({"error": "No file selected. Please choose a WAV audio file."}), 400

    if not allowed_file(f.filename):
        return jsonify({"error": "Unsupported format. Please upload a .wav file."}), 400

    if not model_exists():
        return jsonify({"error": "Model not trained yet. Please train the model first."}), 400

    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        f.save(tmp.name)
        tmp.close()

        from predict import predict_language
        result = predict_language(tmp.name)
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
                # Extract last meaningful line
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
    print("\n  Voice Language Identification System")
    print("  " + "-" * 36)
    print("  Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
