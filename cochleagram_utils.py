import os
import librosa
import numpy as np
from lyon.calc import LyonCalc

from convert_to_wav import sph_to_wav


def compute_cochleagram(audio_path: str, decimation_factor: int = 64) -> np.ndarray:
    """Load an audio file and compute its cochleagram."""

    y, sr = librosa.load(audio_path, sr=None)
    y = y.astype(np.float64)  # LyonCalc expects float64 input

    calc = LyonCalc()
    coch = calc.lyon_passive_ear(
        y,
        int(sr),
        decimation_factor=decimation_factor,
        step_factor=0.195,  # 微調整して100chにした
    )
    return coch


if __name__ == "__main__":
    wave_path = sph_to_wav("audio/train/f1/00f1set0.sph")
    coch = compute_cochleagram(wave_path)
    print(f"Shape of cochleagram: {coch.shape}")

    os.remove(wave_path)
