from src.config.definitions import ROOT_DIR
from src.utils.audio_cleaning import (
    is_audio_file,
    organize_input_samples,
    rename_input_samples,
    clean_audio,
)
from src.utils.smart_splitting import process_split
from src.utils.transcribe import transcribe_all
import os
import argparse
import re


def generate_data(**kwargs):
    folder = kwargs["input_folder"]
    overwrite = kwargs["overwrite"]
    organize_input_samples(folder)
    speaker_name = os.path.basename(folder)
    rename_input_samples(speaker_name, folder)
    wavs_path = os.path.join(ROOT_DIR, "./dataset/{0}/wavs".format(speaker_name))
    generated_files = os.path.exists(wavs_path) and set(
        [re.search(r"(.*?)_[0-9]+", f).group() for f in os.listdir(wavs_path)]
    )
    raw_folder = os.path.join(folder, "raw")
    for file in os.listdir(raw_folder):
        if not is_audio_file(file):
            continue
        if (
            not overwrite
            and generated_files
            and os.path.splitext(os.path.basename(file))[0] in generated_files
        ):
            continue
        path = os.path.join(raw_folder, file)
        print("cleaning audio file")
        clean_path = clean_audio(path)

        print("splitting file...")
        split_path = process_split(clean_path)

        print("transcribing file segments...")
        transcribe_all(split_path, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for generating dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input_folder", help="input audio folder")
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        help="overwrite any duplicates in dataset",
    )
    args = parser.parse_args()

    full_path = os.path.abspath(os.path.join(ROOT_DIR, "samples", args.input_folder))

    if args.input_folder and os.path.exists(full_path):
        generate_data(input_folder=full_path, overwrite=args.overwrite)
    else:
        print("Invalid input file")
