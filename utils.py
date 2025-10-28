import sys


def get_audio_path() -> str:
    if len(sys.argv) < 2:
        print("Usage: *.py <audio_file>")
        sys.exit(1)

    return sys.argv[1]
