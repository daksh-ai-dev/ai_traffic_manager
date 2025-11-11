# siren.py
import soundfile as sf
import numpy as np
import io

def detect_siren_from_audio_bytes(data_bytes):
    """Return (detected:bool, score:float)."""
    try:
        audio, sr = sf.read(io.BytesIO(data_bytes))
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        # compute spectral energy ratio in siren band
        S = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
        band = (freqs > 500) & (freqs < 2000)
        if not band.any():
            return False, 0.0
        band_energy = S[band].mean()
        total_energy = S.mean() + 1e-9
        ratio = float(band_energy / total_energy)
        return (ratio > 0.08), ratio
    except Exception as e:
        print("[WARN] Siren detection failed:", e)
        return False, 0.0
