#!/usr/bin/env python3
"""Train a single-speaker Coqui VITS model on precomputed Persian phonemes."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import soundfile as sf
import torch
from scipy.signal import resample_poly
from trainer import Trainer, TrainerArgs

from persian_g2p_vits.coqui_compat import prepare_coqui_transformers_compatibility

prepare_coqui_transformers_compatibility()

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import dataset as tts_dataset
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models import vits as tts_vits
from TTS.tts.models.vits import CharactersConfig, Vits


def _wav_frame_count(audio_path: str | Path) -> int:
    """Read WAV metadata without TorchCodec/FFmpeg.

    Coqui routes file-length discovery through TorchCodec on PyTorch 2.9+.
    The package needs a system FFmpeg shared library on macOS, although the
    PCM WAV files produced by ``prepare_dataset.py`` can be read directly by
    SoundFile.  Keeping this local override makes the training recipe portable.
    """
    return int(sf.info(str(audio_path)).frames)


# Patch only Coqui's metadata probe; audio waveforms are still loaded through
# its normal AudioProcessor path.
tts_dataset.get_audio_size = _wav_frame_count


def configure_wav_loader(target_sample_rate: int) -> None:
    """Use SoundFile for VITS batch loading and resample source WAVs safely."""

    def load_audio_with_soundfile(audio_path: str | Path) -> tuple[torch.Tensor, int]:
        samples, source_sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        mono = samples.mean(axis=1)
        if source_sample_rate != target_sample_rate:
            divisor = math.gcd(source_sample_rate, target_sample_rate)
            mono = resample_poly(
                mono,
                target_sample_rate // divisor,
                source_sample_rate // divisor,
            ).astype("float32", copy=False)
        return torch.from_numpy(mono.copy()).unsqueeze(0), target_sample_rate

    tts_vits.load_audio = load_audio_with_soundfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/manatts-g2p"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--run-name", default="persian-g2p-vits")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    # A zero-worker default also keeps the SoundFile loader in this process.
    # It avoids macOS spawn workers reverting to PyTorch's TorchCodec loader.
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-samples", type=int, default=25)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=15.0)
    return parser.parse_args()


def read_phoneme_texts(metadata_path: Path) -> list[str]:
    texts: list[str] = []
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        fields = line.split("|")
        if len(fields) >= 3 and fields[2].strip():
            texts.append(fields[2].strip())
    if len(texts) < 2:
        raise RuntimeError(f"Need at least two valid rows in {metadata_path}")
    return texts


def main() -> None:
    args = parse_args()
    configure_wav_loader(args.sample_rate)
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    metadata_path = data_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Run scripts/prepare_dataset.py first: {metadata_path} is missing")
    output_dir.mkdir(parents=True, exist_ok=True)

    phoneme_texts = read_phoneme_texts(metadata_path)
    characters = "".join(sorted(set("".join(phoneme_texts)) - {" "}))
    character_config = CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        characters=characters,
        punctuations=" ",
        pad="<PAD>",
        bos="<BOS>",
        eos="<EOS>",
        blank="<BLNK>",
        phonemes=None,
    )
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        dataset_name="ManaTTS-GE2PE",
        meta_file_train="metadata.csv",
        path=str(data_dir),
        language="fa",
    )
    audio_config = BaseAudioConfig(
        sample_rate=args.sample_rate,
        win_length=1024,
        hop_length=256,
        num_mels=80,
        mel_fmin=0,
        mel_fmax=None,
    )
    eval_count = min(args.eval_samples, max(1, len(phoneme_texts) - 1))
    # Coqui treats values in (0, 1] as a proportion and values >1 as an
    # absolute count. Preserve one training row even in a tiny smoke test.
    eval_size = eval_count if eval_count > 1 else eval_count / len(phoneme_texts)
    config = VitsConfig(
        audio=audio_config,
        run_name=args.run_name,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        batch_group_size=5,
        num_loader_workers=args.num_workers,
        num_eval_loader_workers=args.num_workers,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=args.epochs,
        text_cleaner=None,
        # GE2PE has already converted every input to phonemes. Coqui must not
        # run a second phonemizer over those strings.
        use_phonemes=False,
        compute_input_seq_cache=True,
        characters=character_config,
        test_sentences=phoneme_texts[:3],
        print_step=25,
        print_eval=True,
        mixed_precision=torch.cuda.is_available(),
        min_audio_len=int(args.min_duration * args.sample_rate),
        max_audio_len=int(args.max_duration * args.sample_rate),
        output_path=str(output_dir),
        datasets=[dataset_config],
        cudnn_benchmark=torch.cuda.is_available(),
    )
    train_samples, eval_samples = load_tts_samples(
        # coqui-tts 0.27.x accepts the dataset list rather than VitsConfig.
        config.datasets,
        eval_split=True,
        eval_split_size=eval_size,
    )
    print(f"Training samples: {len(train_samples)}")
    print(f"Evaluation samples: {len(eval_samples)}")
    print(f"Phoneme character inventory ({len(characters)}): {characters}")

    # ``Vits(config)`` creates only the neural network.  Training also needs
    # Coqui's audio processor and tokenizer, which ``init_from_config`` builds
    # from the character inventory above.
    model = Vits.init_from_config(config, samples=train_samples + eval_samples)
    trainer = Trainer(
        TrainerArgs(),
        config,
        str(output_dir),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
