import glob
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os


def match_target_amplitude(aChunk, target_dBFS):
    """Normalize given audio chunk"""
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)


def split(filepath, min_silence=1000):
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
    wavs_path = "./dataset/{0}/wavs".format(base_name)
    files = glob.glob("{0}/*".format(wavs_path))
    for f in files:
        os.remove(f)

    for i, chunk in enumerate(chunks):
        # Create a silence chunk that's 0.5 seconds (or 500 ms) long for padding.
        silence_chunk = AudioSegment.silent(duration=200)

        # Add the padding chunk to beginning and end of the entire chunk.
        audio_chunk = silence_chunk + chunk + silence_chunk

        # Normalize the entire chunk.
        normalized_chunk: AudioSegment = match_target_amplitude(audio_chunk, -20.0)

        # Export the audio chunk with new bitrate.
        print("Exporting {0} chunk{1}.wav.".format(base_name, i))
        wavPath = "./dataset/{0}/wavs/{0}_{1}.wav".format(base_name, i)
        if not os.path.exists(wavPath):
            os.makedirs(os.path.dirname(wavPath), exist_ok=True)
            open(wavPath, "w+").close()

        normalized_chunk.export(
            wavPath,
            bitrate="16k",
            format="wav",
        )
    return chunks
