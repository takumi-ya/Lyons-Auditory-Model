from gtts import gTTS

tts0 = gTTS("zero", lang="en")
tts0.save("audio/zero.wav")

tts1 = gTTS("one", lang="en")
tts1.save("audio/one.wav")

tts2 = gTTS("two", lang="en")
tts2.save("audio/two.wav")
