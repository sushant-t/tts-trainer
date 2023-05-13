## Clean and enhance raw audio
from pydub import AudioSegment
import os
import librosa
import soundfile as sf
import noisereduce as nr


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
    dir_name = os.path.dirname(wav_path)
    file_name = os.path.splitext(os.path.basename(wav_path))[0]
    ext = ".wav"
    output_path = os.path.join(dir_name, file_name + "_trimmed" + ext)

    audio_segment.export(
        output_path,
        format="wav",
    )

    return output_path
