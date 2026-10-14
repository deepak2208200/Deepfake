"""
predict.py  –  Command-line deepfake audio detection.

Usage:
    python predict.py audio.wav
    python predict.py clip1.wav clip2.mp3
    python predict.py --dir ./folder
    python predict.py audio.wav --threshold 0.6
    python predict.py audio.wav --json
"""

import argparse, os, sys, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.model_definitions import DeepfakeEnsemble
from audio_utils.feature_extractor import AudioFeatureExtractor

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "models", "ensemble.pkl")


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        print(f"✗ Model not found at {path}")
        print("  Run:  python run.py train")
        sys.exit(1)
    return DeepfakeEnsemble.load(path)


def predict_file(path, model, extractor, threshold=0.5):
    t0    = time.time()
    feats = extractor.extract_all(path)
    audio_info = getattr(extractor, "last_audio_info", None)
    r     = model.predict_single(feats, threshold=threshold, audio_info=audio_info)
    r["file"]               = os.path.basename(path)
    r["processing_time_s"]  = round(time.time() - t0, 3)
    return r


def print_result(r):
    sym = "🔴 FAKE" if r["label"] == "FAKE" else "🟢 REAL"
    print(f"\n{'─'*55}")
    print(f"  File        : {r['file']}")
    print(f"  Result      : {sym}  (risk: {r['risk_level']})")
    if "content_type" in r:
        print(f"  Content     : {r['content_type']}")
    print(f"  Fake %      : {r['fake_probability']:.2%}")
    print(f"  Real %      : {r['real_probability']:.2%}")
    print(f"  Votes       : {r['model_votes']}")
    print(f"  Time        : {r['processing_time_s']} s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files",       nargs="*")
    p.add_argument("--dir",       default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--json",      action="store_true")
    p.add_argument("--model",     default=MODEL_PATH)
    args = p.parse_args()

    model     = load_model(args.model)
    extractor = AudioFeatureExtractor()

    paths = list(args.files)
    if args.dir:
        for fn in sorted(os.listdir(args.dir)):
            if fn.lower().endswith((".wav",".flac",".mp3",".ogg")):
                paths.append(os.path.join(args.dir, fn))

    if not paths:
        p.print_help(); sys.exit(0)

    results = []
    for fp in paths:
        if not os.path.exists(fp):
            print(f"✗ Not found: {fp}"); continue
        try:
            r = predict_file(fp, model, extractor, args.threshold)
            results.append(r)
            if not args.json: print_result(r)
        except Exception as exc:
            print(f"✗ Error ({fp}): {exc}")

    if args.json:
        print(json.dumps(results, indent=2))

    if results:
        n_fake = sum(1 for r in results if r["label"] == "FAKE")
        print(f"\n  Summary: {n_fake}/{len(results)} detected as FAKE")


if __name__ == "__main__":
    main()
