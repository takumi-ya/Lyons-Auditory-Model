import os
import sys
import matplotlib.pyplot as plt

from cochleagram_utils import compute_cochleagram


def plot_cochleagram(audio_path: str) -> None:
    coch = compute_cochleagram(audio_path, decimation_factor=64)

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
