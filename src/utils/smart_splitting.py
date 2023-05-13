import glob
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
from src.config.definitions import ROOT_DIR
from src.utils.audio_cleaning import normalize_audio, prepare_chunk_for_training
import re


def split(filepath, min_silence=500):
    sound = AudioSegment.from_file(filepath)
    chunks = split_on_silence(
        sound,
        min_silence_len=min_silence,
        silence_thresh=sound.dBFS - 16,
        keep_silence=250,  # optional
    )

    return chunks


def process_split(filepath):
    chunks = split(filepath)
    base_name = os.path.splitext(os.path.basename(filepath))[0]

    speaker_name = re.search(r"(.*?)_[0-9]+", base_name).groups()[0]
    wavs_path = os.path.join(ROOT_DIR, "./dataset/{0}/wavs".format(speaker_name))
    files = glob.glob("{0}/*".format(wavs_path))
    for f in files:
        os.remove(f)

    for i, chunk in enumerate(chunks):
        # Create a silence chunk that's 0.5 seconds (or 500 ms) long for padding.
        silence_chunk = AudioSegment.silent(duration=200)

        # Add the padding chunk to beginning and end of the entire chunk.
        audio_chunk = silence_chunk + chunk + silence_chunk

        # Normalize the entire chunk.
        normalized_chunk: AudioSegment = normalize_audio(audio_chunk, -20.0)

        # Export the audio chunk with new bitrate.
        print(
            "Exporting speaker: {0}, file: {0} chunk{1}.wav.".format(
                speaker_name, base_name, i
            )
        )
        wavPath = os.path.join(
            ROOT_DIR,
            "./dataset/{0}/wavs/{1}_{2}.wav".format(speaker_name, base_name, i),
        )
        if not os.path.exists(wavPath):
            os.makedirs(os.path.dirname(wavPath), exist_ok=True)
            open(wavPath, "w+").close()

        normalized_chunk = prepare_chunk_for_training(normalized_chunk)
        normalized_chunk.export(
            wavPath,
            format="wav",
        )
    return chunks
