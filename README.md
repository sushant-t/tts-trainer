# TTS Trainer

Train Text-To-Speech models using automated dataset generation techniques, such as smart audio splitting with silence detection, and transcription using Whisper.

The datasets generated are in the LJSpeech single-speaker dataset format: https://keithito.com/LJ-Speech-Dataset

## Requirements

Mamba Package Manager: https://github.com/conda-forge/miniforge#mambaforge

Before running, make sure you go through all of the README files located in this repository. You can find them linked below:

1.  [Diarization Model](src/diarization/models)
2.  [Transcription Model](src/models)
3.  [Input Samples](src/samples)

## Instructions

### Generating Data

`python -m src.bin.generate_data --input_folder <INPUT FOLDER NAME>`

Example: for speaker _LJ001_, run `python -m src.bin.generate_data --input_folder LJ001`

### Training

Information coming soon.
