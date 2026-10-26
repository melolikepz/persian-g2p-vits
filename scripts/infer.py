#!/usr/bin/env python3
"""Synthesize Persian text with Homo-GE2PE followed by a trained VITS model."""

from __future__ import annotations

import argparse
from pathlib import Path

from persian_g2p_vits import DEFAULT_G2P_MODEL, PersianG2P
from persian_g2p_vits.coqui_compat import prepare_coqui_transformers_compatibility
from persian_g2p_vits.g2p import default_device

prepare_coqui_transformers_compatibility()

from TTS.api import TTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help="Persian text to synthesize")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/sample.wav"))
    parser.add_argument("--models-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--g2p-model", default=DEFAULT_G2P_MODEL)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    return parser.parse_args()


def newest_checkpoint(models_dir: Path) -> Path:
    preferred = list(models_dir.rglob("best_model*.pth"))
    candidates = preferred or list(models_dir.rglob("checkpoint_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No VITS checkpoint found under {models_dir.resolve()}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> None:
    args = parse_args()
    device = args.device or default_device()
    checkpoint = args.checkpoint.resolve() if args.checkpoint else newest_checkpoint(args.models_dir)
    config = args.config.resolve() if args.config else checkpoint.parent / "config.json"
    if not config.exists():
        raise FileNotFoundError(f"VITS config not found: {config}")

    print(f"Running Persian G2P on: {args.text}")
    phonemes = PersianG2P(args.g2p_model, device=device).phonemize(args.text)
    print(f"Phonemes: {phonemes}")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tts = TTS(
        model_path=str(checkpoint),
        config_path=str(config),
        progress_bar=False,
    ).to(device)
    tts.tts_to_file(text=phonemes, file_path=str(output), split_sentences=False)
    print(f"Audio written to: {output}")


if __name__ == "__main__":
    main()
