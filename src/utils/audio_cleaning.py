## Clean and enhance raw audio
from pydub import AudioSegment
import os
import librosa
import soundfile as sf
import noisereduce as nr
import re


def convert_to_wav(audio_path):
    sound = AudioSegment.from_file(audio_path)
    dir_name = os.path.dirname(audio_path)
    file_name = os.path.splitext(os.path.basename(audio_path))[0]
    ext = ".wav"

    normalized_chunk = sound.set_frame_rate(22050)
    normalized_chunk = sound.set_channels(1)
    normalized_chunk.export(
        os.path.join(dir_name, file_name + ext),
        bitrate="16k",
        format="wav",
    )


def organize_input_samples(input_folder):
    ### we want to put raw audio files in raw folder
    raw_folder = os.path.join(input_folder, "raw")
    if not os.path.exists(raw_folder):
        os.makedirs(raw_folder)
    for f in os.listdir(input_folder):
        src_path = os.path.join(input_folder, f)
        dst_path = os.path.join(raw_folder, f)
        if os.path.isfile(os.path.join(input_folder, f)) and re.search(
            r"\.(?:wav|mp3|aiff|ogg)$", f
        ):
            os.rename(src_path, dst_path)


def rename_input_samples(speaker_name, input_folder):
    raw_folder = os.path.join(input_folder, "raw")
    count = 1
    all_files = os.listdir(raw_folder)
    for f in all_files:
        src_path = os.path.join(raw_folder, f)
        if re.search(rf"{speaker_name}_[0-9]+", f):
            continue
        dst_path = os.path.join(
            raw_folder, f"{speaker_name}_{count}{os.path.splitext(f)[1]}"
        )
        while os.path.exists(dst_path):
            count += 1
            dst_path = os.path.join(
                raw_folder, f"{speaker_name}_{count}{os.path.splitext(f)[1]}"
            )
        os.rename(src_path, dst_path)


def filter_noise(audio_path):
    data, sampling_rate = librosa.load(audio_path)
    reduced_noise = nr.reduce_noise(y=data, sr=sampling_rate, prop_decrease=0.7)

    dir_name = os.path.dirname(audio_path)
    file_name = os.path.splitext(os.path.basename(audio_path))[0]
    ext = ".wav"

    output_path = os.path.join(dir_name, file_name + ext)
    sf.write(output_path, reduced_noise, samplerate=22050, subtype="PCM_16")

    return output_path


def trim_audio_from_diarization(audio, speaker_turns):
    out = AudioSegment.empty()
    for turn in speaker_turns:
        clip = audio[turn["start"] * 1000 : turn["end"] * 1000]
        out = out.append(clip, crossfade=0)

    print("Raw Audio vs. Trimmed Audio:", audio.duration_seconds, out.duration_seconds)

    return out


def normalize_audio(aChunk, target_dBFS):
    """Normalize given audio chunk"""
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)


def prepare_chunk_for_training(audio_segment: AudioSegment):
    audio_segment = audio_segment.set_frame_rate(22050)
    audio_segment = audio_segment.set_channels(1)
    return audio_segment


def save_trimmed_audio(wav_path, audio_segment):
    speaker_dir = os.path.abspath(os.path.join(wav_path, "../.."))
    trimmed_folder = os.path.join(speaker_dir, "trimmed")
    if not os.path.exists(trimmed_folder):
        os.makedirs(trimmed_folder)
    file_name = os.path.splitext(os.path.basename(wav_path))[0]
    ext = ".wav"
    output_path = os.path.join(trimmed_folder, file_name + ext)

    audio_segment.export(
        output_path,
        format="wav",
    )

    return output_path
