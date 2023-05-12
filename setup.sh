mamba create -n tts-trainer
eval "$(conda shell.bash hook)"
conda activate tts-trainer
mamba install -y python=3.10
mamba install -y pip
pip install --upgrade --force-reinstall -r requirements.txt