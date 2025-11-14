import os
import subprocess


def sph_to_wav(sph_path: str) -> str:
    """
    Convert .sph file to .wav using 'sox' command.
    Returns path to the generated .wav file.
    """
    wav_path = sph_path[:-4] + ".wav"
    if not os.path.exists(wav_path):
        # Run sox only if wac does not exist
        subprocess.run(["sox", sph_path, wav_path], check=True)
    return wav_path
