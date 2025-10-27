import os
import sys
import librosa
import matplotlib.pyplot as plt
import numpy as np


from lyon.calc import LyonCalc


def plot_cochleagram(audio_path: str) -> None:
    y, sr = librosa.load(audio_path, sr=None)

    y = y.astype(np.float64)  # LyonCalc expects float64 input
    calc = LyonCalc()
    coch = calc.lyon_passive_ear(y, int(sr), decimation_factor=64)

    plt.figure(figsize=(10, 4))
    plt.imshow(coch.T, aspect="auto", origin="lower", cmap="magma")
    plt.title("Cochleagram (Lyon auditory model)")
    plt.xlabel("Time (frames)")
    plt.ylabel("Cochlear channels (low→high freq)")
    plt.colorbar(label="Activation")
    plt.tight_layout()

    base = os.path.splitext(os.path.basename(audio_path))[0]
    output_filename = f"cochleagram_{base}.png"
    save_path = os.path.join("graph", output_filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Cochleagram saved to {save_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: plot.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]
    plot_cochleagram(audio_file)
