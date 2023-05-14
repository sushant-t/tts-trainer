from pyannote.audio import Pipeline
from pyannote.core import Annotation
import os
import yaml
from collections import Counter
from src.config.definitions import ROOT_DIR
from pydub import AudioSegment
import torch
from itertools import groupby

YAML_PATH = os.path.join(ROOT_DIR, "diarization/pipelines/speaker_diarization.yaml")
MODEL_PATH = os.path.join(ROOT_DIR, "diarization/models/pytorch_model.bin")


def update_diarization_yaml():
    file = open(YAML_PATH, "r+")
    data_map = yaml.safe_load(file)
    data_map["pipeline"]["params"]["segmentation"] = MODEL_PATH
    file = open(YAML_PATH, "w")
    file.write(yaml.dump(data_map, default_flow_style=False))


def diarize_audio(audio_path):
    update_diarization_yaml()
    diarize_pipeline = Pipeline.from_pretrained(YAML_PATH)
    ### push inference to GPU if available
    if torch.cuda.is_available():
        diarize_pipeline = diarize_pipeline.to(0)

    dia = diarize_pipeline(audio_path)
    assert isinstance(dia, Annotation)

    speaker_turns = []
    for speech_turn, _, speaker in dia.itertracks(yield_label=True):
        speaker_turns.append(
            {"start": speech_turn.start, "end": speech_turn.end, "speaker": speaker}
        )

    return speaker_turns


def get_main_speaker(speaker_turns):
    speaker_intervals = [
        (turn["speaker"], turn["end"] - turn["start"]) for turn in speaker_turns
    ]

    freq = []
    for i, group in groupby(sorted(speaker_intervals), key=lambda x: x[0]):
        freq.append((i, sum(v[1] for v in group)))

    return sorted(freq, reverse=True, key=lambda f: f[1])[0][0]


def remove_other_speakers(speaker_turns):
    # make sure majority speaker is obvious, ties are not handled
    for i in range(0, len(speaker_turns) - 1):
        if speaker_turns[i + 1]["start"] < speaker_turns[i]["end"]:  # overlaps
            speaker_turns[i]["end"] = speaker_turns[i + 1]["start"]

    main_speaker = get_main_speaker(speaker_turns)
    speaker_turns = list(
        filter(lambda turn: turn["speaker"] == main_speaker, speaker_turns)
    )

    print(speaker_turns)
    return speaker_turns


def generate_speaker_timestamps(audio_path):
    speaker_turns = diarize_audio(audio_path)
    filtered_speaker_turns = remove_other_speakers(speaker_turns)
    return filtered_speaker_turns
