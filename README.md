# Deepfake Audio Detector
**VTU Project — A Deep Learning-Based Framework for Voice Spoofing and Deepfake Detection**

City Engineering College · CSE VI Semester
Team: Dileep Kumar S · ES Akash · Hemanth G C · Darshan S
Guide: Prof. Mahadev Prasad

---

## What This Project Does

Detects whether an audio clip is **real human speech** or **AI-generated / deepfake audio**
using an ensemble of three machine-learning classifiers trained on 285 acoustic features
extracted from each audio file.

**Detection pipeline:**
1. Load & resample to 16 kHz mono
2. Voice Activity Detection (VAD) — silence removal
3. Noise reduction (spectral denoising)
4. Feature extraction: MFCC + Δ + ΔΔ, spectral, prosodic, GAN-artifact features
5. Ensemble prediction: Gradient Boosting (45%) + Random Forest (35%) + SVM (20%)
6. Risk assessment: VERY LOW / LOW / MEDIUM / HIGH

---

## Quick Start (3 steps)

### Step 1 — Install Python packages
Open a terminal / command prompt **inside the `deepfake_detector` folder** and run:

```bash
pip install -r requirements.txt
```

If you get permission errors on Windows, use:
```bash
pip install -r requirements.txt --user
```

### Step 2 — Train the model (first time only)
```bash
python run.py train
```
This takes about 60 seconds and creates `models/ensemble.pkl`.
You only need to do this once.

### Step 3 — Start the web interface
```bash
python run.py
```
Then open **http://localhost:5000** in your browser (Chrome / Edge / Firefox).

> **Note for VS Code users:** After running `python run.py`, wait for the line
> `✓  Model ready` to appear, then open the browser manually.
> Do **not** click the browser preview inside VS Code — open a real browser window.

---

## Running from VS Code (Recommended)

1. Open the `deepfake_detector` folder in VS Code
2. Open the integrated terminal (`Ctrl + \``)
3. Run: `python run.py`
4. Open Chrome/Edge and go to: `http://localhost:5000`

---

## Command-Line Usage (No Browser Needed)

```bash
# Analyse a single audio file
python predict.py my_audio.wav

# Analyse multiple files
python predict.py file1.wav file2.mp3 file3.flac

# Analyse all files in a folder
python predict.py --dir ./audio_folder

# Adjust threshold (default 0.5)
python predict.py my_audio.wav --threshold 0.6

# JSON output
python predict.py my_audio.wav --json
```

---

## Retrain on Your Own Audio Files

If you have real and fake audio recordings:

```bash
python run.py train --real_dir ./path/to/real --fake_dir ./path/to/fake
```

Supported formats: `.wav`, `.flac`, `.mp3`, `.ogg`

---

## Project Structure

```
deepfake_detector/
├── run.py                    ← START HERE (main launcher)
├── predict.py                ← CLI prediction
├── requirements.txt          ← Python packages
├── README.md                 ← This file
│
├── models/
│   ├── ensemble.pkl          ← Trained model (created by train_models.py)
│   ├── model_definitions.py  ← GBM + RF + SVM ensemble
│   └── train_models.py       ← Training script
│
├── audio_utils/
│   └── feature_extractor.py  ← Audio preprocessing + feature extraction
│
└── web_app/
    ├── app.py                ← Flask web server
    └── templates/
        └── index.html        ← Web interface
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `localhost refused to connect` | Make sure `python run.py` is still running in terminal |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| `Model not found` | Run `python run.py train` first |
| Blank white page in VS Code preview | Open `http://localhost:5000` in Chrome/Edge instead |
| Port 5000 already in use | Kill the old process or change port in `web_app/app.py` |

---

## Technical Details

### Feature Vector (285 dimensions)
| Block | Features | Description |
|---|---|---|
| MFCC | 240 | 40 MFCCs × (mean + std) × 3 (raw + Δ + ΔΔ) |
| Spectral | 24 | Centroid, bandwidth, rolloff, flatness, ZCR, RMS + chroma |
| Prosodic | 8 | F0 statistics, voiced ratio, tempo |
| Artifact | 7 | HF energy ratio, phase discontinuities, HNR, mel entropy |
| Padding | 6 | Residual spectral stats |

### Ensemble Weights
- Gradient Boosting Classifier: **45%**
- Random Forest Classifier: **35%**
- Calibrated SVM: **20%**

### Requirements
- Python 3.9 or newer
- 8 GB RAM recommended
- No GPU required
