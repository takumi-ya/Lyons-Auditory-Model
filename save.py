import os
import numpy as np
from utils import get_audio_path
from cochleagram_utils import compute_cochleagram


def save_npy(audio_path: str):
    coch = compute_cochleagram(audio_path, decimation_factor=64)

    base = os.path.splitext(os.path.basename(audio_path))[0]
    output_filename = f"cochleagram_{base}.npy"
    save_path = os.path.join("npy", output_filename)

    np.save(save_path, coch)
    print(f"Cochleagram saved to {save_path}")


if __name__ == "__main__":
    audio_path = get_audio_path()
    save_npy(audio_path)
