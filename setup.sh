mamba create -n tts-trainer
eval "$(conda shell.bash hook)"
conda activate tts-trainer
mamba install -y python=3.10
mamba install -y pip
mamba install -y pandas
mamba install -y numpy=1.23.1
mamba install -y pydub
pip install TTS
pip install -U openai-whisper
