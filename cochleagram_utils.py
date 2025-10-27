import librosa
import numpy as np
from lyon.calc import LyonCalc


def compute_cochleagram(audio_path: str, decimation_factor: int = 64) -> np.ndarray:
    """Load an audio file and compute its cochleagram."""

    y, sr = librosa.load(audio_path, sr=None)
    y = y.astype(np.float64)  # LyonCalc expects float64 input

    calc = LyonCalc()
    coch = calc.lyon_passive_ear(y, int(sr), decimation_factor=decimation_factor)
    return coch
