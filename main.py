import librosa
import numpy as np

# Lyon モデルの実装をインポート
# 例えば pip install lyon でインストール可能なパッケージが存在します（GitHub “sciforce/lyon”） :contentReference[oaicite:4]{index=4}
from lyon.calc import LyonCalc


def preprocess_with_lyon(wav_path: str, decimation_factor: int = 64) -> np.ndarray:
    """
    Load a wav file, run Lyon's auditory model, and return the cochleagram.
    Parameters:
        wav_path: path to the .wav file
        decimation_factor: integer factor to decimate time axis (for computational cost)
    Returns:
        coch: np.ndarray of shape (T’, N_channels)
            where T’ ≈ (num_samples / decimation_factor), N_channels ≈ 86 (in default)
    """
    # 1. 音声読み込み
    waveform, sr = librosa.load(wav_path, sr=None)  # sr=Noneで元レートを保持
    print(f"Loaded waveform: length={len(waveform)}, sr={sr}")

    # 2. Lyon モデルの計算
    calc = LyonCalc()
    coch = calc.lyon_passive_ear(waveform, int(sr), decimation_factor)
    # coch の形状例： (T’, 86) など :contentReference[oaicite:5]{index=5}
    print(f"Cochleagram shape: {coch.shape}")

    # 3. 必要に応じて正規化や切り出しなどを実施
    # 例：チャネルごと平均0／分散1 にする
    coch_norm = (coch - np.mean(coch, axis=0, keepdims=True)) / (
        np.std(coch, axis=0, keepdims=True) + 1e-9
    )

    return coch_norm


if __name__ == "__main__":
    wav_path = "audio/zero.wav"
    coch_features = preprocess_with_lyon(wav_path, decimation_factor=64)
    # 以降、coch_features をリザバーコンピュータの入力として使う
    # 例：各時刻 t におけるベクトル u_t = coch_features[t, :]
    print("Done preprocessing.")
