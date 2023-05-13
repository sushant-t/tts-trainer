import os

# Trainer: Where the ✨️ happens.
# TrainingArgs: Defines the set of arguments of the Trainer.
from trainer import Trainer, TrainerArgs

# GlowTTSConfig: all model related values for training, validating and testing.
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.config import load_config

# BaseDatasetConfig: defines name, formatter and path of the dataset.
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from src.config.definitions import ROOT_DIR
import argparse

output_path = os.path.join(ROOT_DIR, "dataset")
models_path = os.path.join(ROOT_DIR, "models")

if not os.path.exists(output_path):
    os.makedirs(output_path)


def train(speaker_folder, data_folder):
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train="metadata.csv",
        path=os.path.join(output_path, f"{speaker_folder}/"),
    )

    config_path = os.path.join(
        data_folder if data_folder else ROOT_DIR,
        "/models/tts_models--en--ljspeech--glow-tts/config.json",
    )

    config = load_config(config_path)
    config.update(
        {
            "num_loader_workers": 0,
            "num_eval_loader_workers": 0,
            "output_path": output_path,
            "phoneme_cache_path": os.path.join(output_path, "phoneme_cache"),
            "eval_split_size": 0.027,
            "datasets": [dataset_config],
            "run_eval": False,
            "epochs": 600,
        }
    )

    ap = AudioProcessor.init_from_config(config)
    # Modify sample rate if for a custom audio dataset:

    tokenizer, config = TTSTokenizer.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    model = GlowTTS(config, ap, tokenizer, speaker_manager=None)
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for generating dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-s", "--speaker", help="input speaker (underscores)")
    parser.add_argument("-d", "--data_folder", help="data folder")
    args = parser.parse_args()
    if args.speaker and args.data_folder and os.path.exists(args.data_folder):
        train(args.speaker, args.data_folder)
    else:
        print("Invalid input data")
