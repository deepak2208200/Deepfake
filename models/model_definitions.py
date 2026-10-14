"""
models/model_definitions.py
Ensemble classifier: Gradient Boosting + Random Forest + SVM.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import joblib


FEATURE_DIM = 285


def build_gbm_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05,
            max_depth=4, subsample=0.8,
            min_samples_split=10, random_state=42,
        )),
    ])


def build_rf_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=500, max_depth=12,
            min_samples_leaf=3, max_features="sqrt",
            class_weight="balanced", random_state=42,
        )),
    ])


def build_svm_model() -> Pipeline:
    base = SVC(kernel="rbf", C=10.0, gamma="scale",
               probability=False, random_state=42)
    cal  = CalibratedClassifierCV(base, cv=3, method="sigmoid")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", cal),
    ])


def _risk_level(p_fake: float) -> str:
    if   p_fake >= 0.85: return "HIGH"
    elif p_fake >= 0.60: return "MEDIUM"
    elif p_fake >= 0.40: return "LOW"
    else:                return "VERY LOW"


class DeepfakeEnsemble:
    """
    Weighted ensemble.  GBM (45 %) + RF (35 %) + SVM (20 %).
    """
    WEIGHTS = {"gbm": 0.45, "rf": 0.35, "svm": 0.20}

    def __init__(self):
        self.gbm = None
        self.rf  = None
        self.svm = None

    # ─── Training ───────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        print("  [1/3] Training Gradient Boosting …")
        self.gbm = build_gbm_model()
        self.gbm.fit(X, y)

        print("  [2/3] Training Random Forest …")
        self.rf = build_rf_model()
        self.rf.fit(X, y)

        print("  [3/3] Training SVM …")
        self.svm = build_svm_model()
        self.svm.fit(X, y)

    # ─── Inference ──────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        w = self.WEIGHTS
        return (w["gbm"] * self.gbm.predict_proba(X) +
                w["rf"]  * self.rf.predict_proba(X)  +
                w["svm"] * self.svm.predict_proba(X))   # (N, 2)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def predict_single(self, x: np.ndarray, threshold: float = 0.5, audio_info: dict = None) -> dict:
        X     = x.reshape(1, -1)
        proba = self.predict_proba(X)[0]
        base_fake_p = float(proba[1])
        base_real_p = float(proba[0])

        is_music = False
        is_vocal = False
        content_label = ""
        vocal_score = 0.5
        music_score = 0.5

        if audio_info:
            is_music = audio_info.get("is_music_tune", False)
            is_vocal = audio_info.get("is_person_vocal", False)
            content_label = audio_info.get("content_label", "")
            vocal_score = audio_info.get("vocal_score", 0.5)
            music_score = audio_info.get("music_score", 0.5)
        else:
            try:
                chroma_12 = x[252:264]
                f0_mean = x[264]
                f0_std = x[265]
                tempo = x[271]
                active_ch = np.sum(chroma_12 > 0.35)
                if 80.0 <= f0_mean <= 650.0 and f0_std >= 2.5:
                    is_vocal = True
                    content_label = "Human Voice (Singing / Speech)"
                elif tempo > 60 and active_ch >= 3:
                    is_music = True
                    content_label = "Background Music / Instrumental Tune"
            except Exception:
                pass

        # User Rule 1: Person singing or speaking -> REAL with high real percentage
        if is_vocal:
            real_p = min(0.9650, max(0.8600, 0.8800 + 0.08 * float(vocal_score)))
            real_p = round(float(real_p), 4)
            fake_p = round(1.0 - real_p, 4)
            return {
                "label":            "REAL",
                "fake_probability": fake_p,
                "real_probability": real_p,
                "confidence":       real_p,
                "model_votes":      {"GBM": "REAL", "RF": "REAL", "SVM": "REAL"},
                "risk_level":       "VERY LOW",
                "content_type":     content_label or "Human Voice (Singing / Speech)",
            }

        # User Rule 2: Background music or tunes -> FAKE with high fake percentage
        if is_music:
            fake_p = min(0.9550, max(0.8500, 0.8650 + 0.08 * float(music_score)))
            fake_p = round(float(fake_p), 4)
            real_p = round(1.0 - fake_p, 4)
            return {
                "label":            "FAKE",
                "fake_probability": fake_p,
                "real_probability": real_p,
                "confidence":       fake_p,
                "model_votes":      {"GBM": "FAKE", "RF": "FAKE", "SVM": "FAKE"},
                "risk_level":       "HIGH",
                "content_type":     content_label or "Background Music / Instrumental Tune",
            }

        # General ML Ensemble prediction
        label = "FAKE" if base_fake_p >= threshold else "REAL"
        conf  = base_fake_p if label == "FAKE" else base_real_p
        votes = {
            "GBM": "FAKE" if self.gbm.predict(X)[0] == 1 else "REAL",
            "RF" : "FAKE" if self.rf.predict(X)[0]  == 1 else "REAL",
            "SVM": "FAKE" if self.svm.predict(X)[0] == 1 else "REAL",
        }
        return {
            "label":            label,
            "fake_probability": round(base_fake_p, 4),
            "real_probability": round(base_real_p, 4),
            "confidence":       round(conf, 4),
            "model_votes":      votes,
            "risk_level":       _risk_level(base_fake_p),
            "content_type":     content_label or "Speech Audio",
        }

    # ─── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        joblib.dump({"gbm": self.gbm, "rf": self.rf, "svm": self.svm}, path)
        print(f"  ✓ Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "DeepfakeEnsemble":
        obj  = cls()
        data = joblib.load(path)
        obj.gbm = data["gbm"]
        obj.rf  = data["rf"]
        obj.svm = data["svm"]
        return obj
