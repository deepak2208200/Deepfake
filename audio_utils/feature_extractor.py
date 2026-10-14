"""
audio_utils/feature_extractor.py
Audio preprocessing and feature extraction pipeline.

Pipeline:
  1. Load & resample to 16 kHz mono
  2. Voice Activity Detection (VAD) – silence removal
  3. Noise reduction
  4. Feature extraction: MFCC + delta + delta2 (240-D)
                         Spectral stats               (24-D)
                         Prosodic / F0                (8-D)
                         GAN-artifact indicators      (7-D)
                         Residual padding             (6-D)
                         ─────────────────────────────────
                         Total                       (285-D)
"""

import numpy as np
import librosa
import warnings
warnings.filterwarnings("ignore")

try:
    import noisereduce as nr
    _HAS_NR = True
except ImportError:
    _HAS_NR = False


class AudioFeatureExtractor:
    def __init__(self, sr: int = 16000, n_mfcc: int = 40,
                 hop_length: int = 512, n_fft: int = 2048):
        self.sr          = sr
        self.n_mfcc      = n_mfcc
        self.hop_length  = hop_length
        self.n_fft       = n_fft

    # ─── Loading ────────────────────────────────────────────────────────────

    def load_audio(self, path: str, duration: float = None) -> np.ndarray:
        """Load audio, resample to self.sr, peak-normalise."""
        y, _ = librosa.load(path, sr=self.sr, duration=duration, mono=True)
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak
        return y

    # ─── Preprocessing ──────────────────────────────────────────────────────

    def apply_vad(self, y: np.ndarray, top_db: float = 30.0) -> np.ndarray:
        """Voice Activity Detection: remove leading/trailing silence."""
        intervals = librosa.effects.split(y, top_db=top_db)
        if len(intervals) == 0:
            return y
        parts = [y[s:e] for s, e in intervals]
        return np.concatenate(parts)

    def apply_noise_reduction(self, y: np.ndarray) -> np.ndarray:
        """Spectral noise reduction using noisereduce (if available)."""
        if not _HAS_NR:
            return y
        try:
            return nr.reduce_noise(y=y, sr=self.sr, stationary=False,
                                   prop_decrease=0.75)
        except Exception:
            return y

    def preprocess(self, y: np.ndarray) -> np.ndarray:
        """Full preprocessing: VAD → noise reduction."""
        y = self.apply_vad(y)
        y = self.apply_noise_reduction(y)
        # Normalise again after processing
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak
        return y

    # ─── Audio Content Analysis (Music/Tune vs Singing/Speech) ───────────────

    def analyze_audio_content(self, y: np.ndarray) -> dict:
        """
        Differentiates between:
        1. Background music / instrumental tunes (BGM, tunes, beats, non-vocal audio) -> FAKE
        2. Human voice: person singing or speaking (song or speech) -> REAL
        """
        sr = self.sr
        if len(y) < sr * 0.3:
            return {
                "is_person_vocal": False,
                "is_music_tune": False,
                "vocal_score": 0.5,
                "music_score": 0.5,
                "content_label": "Short Audio",
            }

        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak

        # 1. Harmonic vs Percussive (HPSS)
        try:
            y_harm, y_perc = librosa.effects.hpss(y)
            harm_power = float(np.mean(y_harm ** 2))
            perc_power = float(np.mean(y_perc ** 2))
            total_power = harm_power + perc_power + 1e-9
            perc_ratio = perc_power / total_power
        except Exception:
            perc_ratio = 0.0

        # 2. Formant & Spectral Energy Dominance
        stft = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)

        vocal_mask = (freqs >= 300) & (freqs <= 3400)
        bass_mask = freqs < 220
        high_mask = freqs > 4500

        total_spec_energy = float(stft.sum()) + 1e-9
        vocal_energy = float(stft[vocal_mask].sum())
        bass_energy = float(stft[bass_mask].sum())
        high_energy = float(stft[high_mask].sum())

        vocal_dominance = vocal_energy / total_spec_energy
        bass_ratio = bass_energy / total_spec_energy
        high_ratio = high_energy / total_spec_energy
        non_vocal_ratio = (bass_energy + high_energy) / total_spec_energy

        # 3. Chroma Polyphony (Chords in music vs monophonic vocal line)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
        active_pitches = np.sum(chroma > 0.30, axis=0)
        mean_polyphony = float(np.mean(active_pitches))
        chroma_active_bins = int(np.sum(chroma.mean(axis=1) > 0.22))

        # 4. Rhythm and Beat Regularity
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            tempo_val = float(np.atleast_1d(tempo)[0])
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            beat_pulse = float(np.std(onset_env))
        except Exception:
            tempo_val = 0.0
            beat_pulse = 0.0

        # 5. Formant Dynamics (Human vocal tract creates shifting F1/F2 formants)
        try:
            mfcc_raw = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=self.hop_length)
            formant_dyn = float(np.mean(np.std(mfcc_raw[1:5], axis=1)))
        except Exception:
            formant_dyn = 0.0

        # 6. Consonant / Phoneme Dynamics (Unvoiced /s/, /t/, /k/ fricatives in speech/singing)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)[0]
        zcr_std = float(np.std(zcr))
        zcr_p90 = float(np.percentile(zcr, 90))

        # 7. Pitch (F0) Tracking in Human Vocal Range (80 Hz - 650 Hz)
        try:
            f0, voiced_flag, _ = librosa.pyin(
                y,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr,
                frame_length=self.n_fft,
                hop_length=self.hop_length,
            )
            f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([])
            voiced_ratio = float(np.mean(voiced_flag)) if voiced_flag is not None else 0.0
        except Exception:
            f0_clean = np.array([])
            voiced_ratio = 0.0

        has_vocal_f0 = False
        f0_std = 0.0
        f0_mean = 0.0
        if len(f0_clean) > 6:
            f0_mean = float(np.mean(f0_clean))
            f0_std = float(np.std(f0_clean))
            if 80.0 <= f0_mean <= 650.0 and f0_std >= 2.5:
                has_vocal_f0 = True

        # Evidence scoring
        voice_points = 0
        if formant_dyn >= 15.0: voice_points += 3
        if zcr_std >= 0.024 or zcr_p90 >= 0.09: voice_points += 2
        if has_vocal_f0: voice_points += 2
        if vocal_dominance >= 0.45: voice_points += 2
        if mean_polyphony < 2.2: voice_points += 2

        music_points = 0
        if mean_polyphony >= 2.2: music_points += 3
        if chroma_active_bins >= 3: music_points += 2
        if formant_dyn < 14.0: music_points += 2
        if non_vocal_ratio >= 0.30: music_points += 2
        if perc_ratio >= 0.16 and tempo_val > 50: music_points += 2
        if zcr_std < 0.022: music_points += 2

        # ── 1. Check for Human Voice Presence (Speech, Conversation, Singing) ──
        # A human voice speaking or singing produces:
        # - Pitch in human vocal range (80 - 650 Hz)
        # - Vowel-consonant formant shifts (formant_dyn)
        # - Unvoiced consonant friction bursts (zcr_p90 / zcr_std)
        # - Substantial energy in vocal range (300 - 3400 Hz)
        has_human_speech_or_singing = False

        if has_vocal_f0 and (formant_dyn >= 11.5 or zcr_p90 >= 0.055 or vocal_dominance >= 0.35):
            has_human_speech_or_singing = True
        elif vocal_dominance >= 0.48 and formant_dyn >= 12.5 and zcr_p90 >= 0.055:
            has_human_speech_or_singing = True
        elif voice_points >= 4 and voice_points >= music_points:
            has_human_speech_or_singing = True

        # ── 2. Decision Logic ──
        if has_human_speech_or_singing:
            # Human speech, conversation, or singing detected -> REAL
            is_person_vocal = True
            is_music_tune = False
            content_label = "Human Voice (Conversation / Singing / Speech)"
        else:
            # Pure background music, instrumental tunes, beats, or synth -> FAKE
            is_person_vocal = False
            is_music_tune = True
            content_label = "Background Music / Instrumental Tune"

        return {
            "is_person_vocal": is_person_vocal,
            "is_music_tune": is_music_tune,
            "vocal_score": round(float(voice_points / 11.0), 3),
            "music_score": round(float(music_points / 13.0), 3),
            "content_label": content_label,
            "vocal_dominance": round(float(vocal_dominance), 3),
            "non_vocal_ratio": round(float(non_vocal_ratio), 3),
            "mean_polyphony": round(float(mean_polyphony), 2),
            "formant_dyn": round(float(formant_dyn), 2),
            "zcr_std": round(float(zcr_std), 4),
            "f0_mean": round(float(f0_mean), 1),
            "f0_std": round(float(f0_std), 1),
        }

    # ─── Feature extraction ─────────────────────────────────────────────────

    def extract_mfcc_features(self, y: np.ndarray) -> np.ndarray:
        """MFCC + Δ + ΔΔ statistics → 240 features."""
        mfcc   = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc,
                                      hop_length=self.hop_length, n_fft=self.n_fft)
        delta  = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        feats = []
        for m in [mfcc, delta, delta2]:
            feats.append(m.mean(axis=1))   # (40,)
            feats.append(m.std(axis=1))    # (40,)
        return np.concatenate(feats)       # 240

    def extract_spectral_features(self, y: np.ndarray) -> np.ndarray:
        """Spectral + chroma → 24 features."""
        feats = []

        sc = librosa.feature.spectral_centroid(y=y, sr=self.sr,
                                               hop_length=self.hop_length)
        feats += [float(sc.mean()), float(sc.std())]

        sb = librosa.feature.spectral_bandwidth(y=y, sr=self.sr,
                                                hop_length=self.hop_length)
        feats += [float(sb.mean()), float(sb.std())]

        sr_f = librosa.feature.spectral_rolloff(y=y, sr=self.sr,
                                                hop_length=self.hop_length)
        feats += [float(sr_f.mean()), float(sr_f.std())]

        sf = librosa.feature.spectral_flatness(y=y, hop_length=self.hop_length)
        feats += [float(sf.mean()), float(sf.std())]

        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)
        feats += [float(zcr.mean()), float(zcr.std())]

        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)
        feats += [float(rms.mean()), float(rms.std())]

        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr,
                                             hop_length=self.hop_length)
        feats += chroma.mean(axis=1).tolist()   # 12 values

        return np.array(feats, dtype=float)     # 24

    def extract_prosodic_features(self, y: np.ndarray) -> np.ndarray:
        """F0 stats + voiced ratio + tempo → 8 features."""
        feats = []
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sr,
        )
        f0_clean = f0[~np.isnan(f0)]
        if len(f0_clean) > 0:
            feats += [float(f0_clean.mean()), float(f0_clean.std()),
                      float(f0_clean.min()),  float(f0_clean.max()),
                      float(np.percentile(f0_clean, 25)),
                      float(np.percentile(f0_clean, 75))]
        else:
            feats += [0.0] * 6

        feats.append(float(voiced_flag.mean()) if voiced_flag is not None else 0.0)

        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=self.sr)
            feats.append(float(np.atleast_1d(tempo)[0]))
        except Exception:
            feats.append(0.0)

        return np.array(feats, dtype=float)     # 8

    def extract_artifact_features(self, y: np.ndarray) -> np.ndarray:
        """GAN/TTS artifact indicators → 7 features."""
        feats = []
        stft  = np.abs(librosa.stft(y, n_fft=self.n_fft,
                                    hop_length=self.hop_length))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)

        low_e = stft[freqs < 4000].mean()  + 1e-8
        hf_e  = stft[freqs >= 4000].mean() + 1e-8
        vhf_e = stft[freqs >= 7000].mean() + 1e-8
        feats += [float(hf_e / low_e), float(vhf_e / low_e)]

        phase     = np.angle(librosa.stft(y, n_fft=self.n_fft,
                                         hop_length=self.hop_length))
        inst_freq = np.diff(phase, axis=1)
        feats    += [float(inst_freq.mean()), float(inst_freq.std())]

        cepstrum    = np.abs(np.fft.ifft(np.log(stft.mean(axis=1) + 1e-8)))
        pitch_slice = cepstrum[1: max(2, self.sr // 200)]
        feats.append(float(pitch_slice.max()) if len(pitch_slice) > 0 else 0.0)

        harmonic, percussive = librosa.effects.hpss(y)
        hnr = (np.var(harmonic) + 1e-8) / (np.var(percussive) + 1e-8)
        feats.append(float(np.log1p(hnr)))

        mel      = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=64)
        mel_mean = mel.mean(axis=1) + 1e-8
        mel_prob = mel_mean / mel_mean.sum()
        feats.append(float(-np.sum(mel_prob * np.log2(mel_prob + 1e-10))))

        return np.array(feats, dtype=float)     # 7

    # ─── Full pipeline ───────────────────────────────────────────────────────

    def extract_all(self, path: str, duration: float = 10.0) -> np.ndarray:
        """
        Load → preprocess → extract → return 285-D feature vector.
        Raises RuntimeError on failure.
        """
        try:
            y = self.load_audio(path, duration=duration)
            if len(y) < self.sr * 0.5:
                raise ValueError("Audio too short (< 0.5 s)")

            # Analyze audio nature on raw waveform (music/tune vs singing/speech)
            self.last_audio_info = self.analyze_audio_content(y)

            y = self.preprocess(y)

            mfcc_f = self.extract_mfcc_features(y)    # 240
            spec_f = self.extract_spectral_features(y) # 24
            pros_f = self.extract_prosodic_features(y) # 8
            art_f  = self.extract_artifact_features(y) # 7
            pad    = np.zeros(6, dtype=float)           # 6  padding

            vec = np.concatenate([mfcc_f, spec_f, pros_f, art_f, pad])
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
            return vec                                  # 285
        except Exception as exc:
            raise RuntimeError(
                f"Feature extraction failed for {path}: {exc}"
            ) from exc
