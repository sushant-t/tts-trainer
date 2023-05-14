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
    no_overwrite = kwargs["no_overwrite"]
    organize_input_samples(folder)
    speaker_name = os.path.basename(folder)
    rename_input_samples(speaker_name, folder)
    wavs_path = os.path.join(ROOT_DIR, "./dataset/{0}/wavs".format(speaker_name))
    generated_files = set(
        [re.search(r"(.*?)_[0-9]+", f).group() for f in os.listdir(wavs_path)]
    )
    raw_folder = os.path.join(folder, "raw")
    for file in os.listdir(raw_folder):
        if not is_audio_file(file):
            continue
        if (
            no_overwrite
            and os.path.splitext(os.path.basename(file))[0] in generated_files
        ):
            continue
        path = os.path.join(raw_folder, file)
        print("cleaning audio file")
        clean_path = clean_audio(path)

        print("splitting file...")
        split_path = process_split(clean_path)

        print("transcribing file segments...")
        transcribe_all(split_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for generating dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input_folder", help="input audio folder")
    parser.add_argument(
        "--no_overwrite",
        action=argparse.BooleanOptionalAction,
        help="do not overwrite any duplicates in dataset",
    )
    args = parser.parse_args()
    full_path = (
        os.path.join(ROOT_DIR, args.input_folder)
        if os.path.exists(os.path.join(ROOT_DIR, args.input_folder))
        else os.path.exists(os.path.join(os.getcwd(), args.input_folder))
    )
    if args.input_folder and os.path.exists(full_path):
        generate_data(input_folder=args.input_folder, no_overwrite=args.no_overwrite)
    else:
        print("Invalid input file")
