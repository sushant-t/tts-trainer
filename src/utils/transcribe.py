import whisper
import pandas as pd
import os
from src.config.definitions import ROOT_DIR

model = whisper.load_model("base")


def transcribe_file(file_path) -> str:
    result = model.transcribe(
        file_path,
        fp16=False,
        language="English",
    )
    return result["text"]


def transcribe_all(base_name):
    base_path = os.path.join(ROOT_DIR, "./dataset/{0}".format(base_name))
    wavs_path = "{0}/wavs".format(base_path)
    metadata_path = os.path.join(base_path, "metadata.csv")

    columns = ["file_name", "transcription", "normalized_transcription"]
    if os.path.exists(metadata_path):
        metadata = pd.read_csv(metadata_path, sep="|", header=None, names=columns)

    else:
        metadata: pd.DataFrame = pd.DataFrame(columns=columns)

    files = sorted(
        os.listdir(wavs_path),
        key=lambda k: int(
            os.path.splitext(k)[0].split("_")[-1]
        ),  # sort by the chunk number
    )

    for file in files:
        transcription = transcribe_file(os.path.join(wavs_path, file)).strip()
        metadata.loc[len(metadata)] = {
            "file_name": os.path.splitext(file)[0],
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
