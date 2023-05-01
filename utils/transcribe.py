import whisper
from pydub import AudioSegment
import numpy as np
from pydub.playback import play
import pandas as pd
import os

model = whisper.load_model("base")


def transcribe_file(file_path) -> str:
    result = model.transcribe(
        file_path,
        fp16=False,
        language="English",
    )
    return result["text"]


def transcribe_all(base_name):
    base_path = "./dataset/{0}".format(base_name)
    wavs_path = "{0}/wavs".format(base_path)
    metadata: pd.DataFrame = pd.DataFrame(
        columns=["file_name", "transcription", "normalized_transcription"]
    )

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
        os.path.join(base_path, "metadata.csv"),
        header=False,
        sep="|",
        index=False,
        lineterminator="\n",
    )
