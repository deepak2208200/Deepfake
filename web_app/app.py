"""
web_app/app.py  –  Flask web server for the Deepfake Audio Detector.

Run:
    python web_app/app.py          (from project root)
  OR:
    python run.py                  (recommended launcher)
"""

import os
import sys
import tempfile
import time
import json

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from flask import Flask, request, jsonify, render_template, send_from_directory

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_definitions import DeepfakeEnsemble
from audio_utils.feature_extractor import AudioFeatureExtractor

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "ensemble.pkl")

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "templates"),
    static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # 50 MB

ALLOWED_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".mp4", ".mkv", ".webm", ".mov", ".avi"}

_model     = None
_extractor = None


def get_model():
    global _model, _extractor
    if _model is None:
        _model     = DeepfakeEnsemble.load(MODEL_PATH)
        _extractor = AudioFeatureExtractor()
    return _model, _extractor


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/analyse", methods=["POST"])
def analyse():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f   = request.files["file"]
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported format '{ext}'. "
                                  f"Allowed: {', '.join(sorted(ALLOWED_EXT))}"}), 400

    threshold = float(request.form.get("threshold", 0.5))

    try:
        model, _ = get_model()
        extractor = AudioFeatureExtractor()
    except Exception as exc:
        return jsonify({"error": f"Model not loaded: {exc}. "
                                  "Run: python models/train_models.py"}), 500

    # Save upload to a temp file
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        t0     = time.time()
        feats  = extractor.extract_all(tmp_path, duration=10.0)
        audio_info = getattr(extractor, "last_audio_info", None)
        result = model.predict_single(feats, threshold=threshold, audio_info=audio_info)
        result["processing_time_s"] = round(time.time() - t0, 3)
        result["filename"]          = f.filename
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    print("=" * 60)
    print("  Deepfake Audio Detector  –  Web Interface")
    print("  http://localhost:5000")
    print("=" * 60)
    try:
        get_model()
        print("  ✓ Model loaded successfully")
    except Exception as exc:
        print(f"  ✗ {exc}")
        print("    Run:  python models/train_models.py")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=True)
