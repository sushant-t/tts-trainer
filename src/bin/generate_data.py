from src.config.definitions import ROOT_DIR
from src.diarization.diarize import generate_speaker_timestamps
from src.utils.audio_cleaning import (
    filter_noise,
    trim_audio_from_diarization,
    normalize_audio,
    save_trimmed_audio,
)
from src.utils.smart_splitting import process_split
from src.utils.transcribe import transcribe_all
from pydub import AudioSegment
import os
import argparse


def clean_audio(audio_path):
    output_path = filter_noise(audio_path)

    filtered_speaker_turns = generate_speaker_timestamps(output_path)

    audio = AudioSegment.from_file(output_path)
    audio = normalize_audio(audio, -20.0)
    trimmed_output = trim_audio_from_diarization(audio, filtered_speaker_turns)

    trimmed_output_path = save_trimmed_audio(output_path, trimmed_output)

    return trimmed_output_path


def generate_data(path):
    print("cleaning audio file")
    clean_path = clean_audio(path)

    print("splitting file...")
    process_split(clean_path)

    base_name = os.path.splitext(os.path.basename(clean_path))[0]

    print("transcribing file segments...")
    transcribe_all(base_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for generating dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input_file", help="input audio file")
    args = parser.parse_args()
    if args.input_file and os.path.exists(args.input_file):
        generate_data(args.input_file)
    else:
        print("Invalid input file")
