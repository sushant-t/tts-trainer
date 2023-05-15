import whisper
import pandas as pd
import os
from src.config.definitions import ROOT_DIR

whisper_path = os.path.join(ROOT_DIR, "models/base.pt")
if os.path.exists(whisper_path):
    model = whisper.load_model(whisper_path)
else:
    model = whisper.load_model("base")


def transcribe_file(file_path) -> str:
    result = model.transcribe(
        file_path,
        fp16=False,
        language="English",
    )
    return result["text"]


def transcribe_all(wavs_path, raw_file):
    metadata_path = os.path.abspath(os.path.join(wavs_path, "..", "metadata.csv"))

    columns = ["file_name", "transcription", "normalized_transcription"]
    if os.path.exists(metadata_path):
        metadata = pd.read_csv(metadata_path, sep="|", header=None, names=columns)
    else:
        metadata: pd.DataFrame = pd.DataFrame(columns=columns)

    ### we need to make sure we are only transcribing files created from the current raw file
    files = sorted(
        [
            wav
            for wav in os.listdir(wavs_path)
            if wav.startswith(os.path.splitext(raw_file)[0])
        ],
        key=lambda k: int(
            os.path.splitext(k)[0].split("_")[-1]
        ),  # sort by the chunk number
    )

    for file in files:
        transcription = transcribe_file(os.path.join(wavs_path, file)).strip()
        ### remove bad audio files that don't have valid transcription
        if not transcription:  # not valid
            os.remove(os.path.join(wavs_path, file))
            continue
        file_name = os.path.splitext(file)[0]
        data = {
            "file_name": file_name,
            "transcription": transcription,
            "normalized_transcription": transcription,
        }
        if not metadata.loc[metadata["file_name"] == file_name].empty:
            metadata.loc[metadata["file_name"] == file_name, list(data.keys())] = list(
                data.values()
            )
            print(metadata.loc[metadata["file_name"] == file_name])
        else:
            metadata.loc[len(metadata)] = {
                "file_name": file_name,
                "transcription": transcription,
                "normalized_transcription": transcription,
            }
            print(metadata.loc[len(metadata) - 1])

    metadata.to_csv(
        metadata_path,
        header=False,
        sep="|",
        index=False,
        lineterminator="\n",
    )
