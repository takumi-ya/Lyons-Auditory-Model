import os
import matplotlib.pyplot as plt

from convert_to_wav import sph_to_wav
from utils import get_audio_path
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
    save_path = os.path.join("fig", output_filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Cochleagram saved to {save_path}")


def plot_single_channel(audio_path: str, channel_index: int) -> None:
    coch = compute_cochleagram(audio_path, decimation_factor=64)

    # coch の形状確認（転置しているかどうか）
    # あなたの plot では plt.imshow(coch.T) を使っていたので、
    # coch.shape = (channels, time) と仮定します。
    ch_data = coch[channel_index]

    plt.figure(figsize=(10, 4))
    plt.plot(ch_data)
    plt.title(f"Cochleagram Channel {channel_index}")
    plt.xlabel("Time (frames)")
    plt.ylabel("Activation")
    plt.tight_layout()

    base = os.path.splitext(os.path.basename(audio_path))[0]
    output_filename = f"cochleagram_ch{channel_index}_{base}.png"
    save_path = os.path.join("fig", output_filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Single-channel graph saved to {save_path}")


if __name__ == "__main__":
    audio_path = get_audio_path()
    if audio_path[-4:] == ".sph":
        audio_path = sph_to_wav(audio_path)
    plot_cochleagram(audio_path)

    plot_single_channel(audio_path, channel_index=10)
