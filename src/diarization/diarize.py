from pyannote.audio import Pipeline
from pyannote.core import Annotation
import os
import yaml
from collections import Counter
from src.config.definitions import ROOT_DIR
from pydub import AudioSegment
import torch

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
    # for i in range(torch.cuda.device_count()):
    #     diarize_pipeline = diarize_pipeline.to(0)

    dia = diarize_pipeline(audio_path)
    assert isinstance(dia, Annotation)

    speaker_turns = []
    for speech_turn, _, speaker in dia.itertracks(yield_label=True):
        speaker_turns.append(
            {"start": speech_turn.start, "end": speech_turn.end, "speaker": speaker}
        )

    return speaker_turns


def remove_other_speakers(speaker_turns):
    # make sure majority speaker is obvious, ties are not handled
    main_speaker, _ = Counter(map(lambda k: k["speaker"], speaker_turns)).most_common()[
        0
    ]
    for i in range(0, len(speaker_turns) - 1):
        if speaker_turns[i + 1]["start"] < speaker_turns[i]["end"]:  # overlaps
            speaker_turns[i]["end"] = speaker_turns[i + 1]["start"]

    speaker_turns = list(
        filter(lambda turn: turn["speaker"] == main_speaker, speaker_turns)
    )

    print(speaker_turns)
    return speaker_turns


def generate_speaker_timestamps(audio_path):
    speaker_turns = diarize_audio(audio_path)
    filtered_speaker_turns = remove_other_speakers(speaker_turns)
    return filtered_speaker_turns
