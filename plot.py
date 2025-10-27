import librosa
import matplotlib.pyplot as plt

from lyon.calc import LyonCalc

y, sr = librosa.load("audio/zero.wav", sr=None)
calc = LyonCalc()
coch = calc.lyon_passive_ear(y, int(sr), decimation_factor=64)

plt.figure(figsize=(10, 4))
plt.imshow(coch.T, aspect='auto', origin='lower', cmap='magma')
plt.title("Cochleagram (Lyon auditory model)")
plt.xlabel("Time (frames)")
plt.ylabel("Cochlear channels (low→high freq)")
plt.colorbar(label="Activation")
plt.tight_layout()
plt.show()
