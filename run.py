#!/usr/bin/env python3
"""
run.py  –  One-click launcher for the Deepfake Audio Detector.

  python run.py              → start web UI at http://localhost:5000
  python run.py train        → retrain model on synthetic data
  python run.py cli audio.wav  → command-line prediction
"""

import sys
import os
import subprocess

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE  = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(BASE, "models", "ensemble.pkl")


def check_and_train():
    if not os.path.exists(MODEL):
        print("⚙  Model not found — training now (takes ~60 s)…")
        subprocess.check_call(
            [sys.executable,
             os.path.join(BASE, "models", "train_models.py")])
    else:
        print(f"✓  Model ready: {MODEL}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "web"

    if mode == "train":
        subprocess.check_call(
            [sys.executable,
             os.path.join(BASE, "models", "train_models.py")] + sys.argv[2:])

    elif mode == "cli":
        check_and_train()
        subprocess.check_call(
            [sys.executable,
             os.path.join(BASE, "predict.py")] + sys.argv[2:])

    else:   # web (default)
        check_and_train()
        print("\n🌐  http://localhost:5000  –  open this in your browser\n")
        subprocess.check_call(
            [sys.executable,
             os.path.join(BASE, "web_app", "app.py")])


if __name__ == "__main__":
    main()
