#!/usr/bin/env python3
"""Download ManaTTS, run Homo-GE2PE, and create a Coqui manifest."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from datasets import load_dataset

from persian_g2p_vits import DEFAULT_G2P_MODEL, PersianG2P, normalize_persian_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/manatts-g2p"))
    parser.add_argument("--data-files", default="dataset/dataset_part_001.parquet")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=15.0)
    parser.add_argument("--g2p-model", default=DEFAULT_G2P_MODEL)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def audio_array(row: dict[str, Any]) -> tuple[np.ndarray, int]:
    """Extract samples and sample rate from a Hugging Face Audio value."""
    audio = row["audio"]
    fallback_rate = int(row.get("samplerate") or 44100)
    if isinstance(audio, dict):
        sample_rate = int(audio.get("sampling_rate") or fallback_rate)
        if audio.get("array") is not None:
            return np.asarray(audio["array"], dtype=np.float32), sample_rate
        if audio.get("bytes") is not None:
            values, decoded_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            return np.asarray(values, dtype=np.float32), int(decoded_rate)
        if audio.get("path"):
            values, decoded_rate = sf.read(audio["path"], dtype="float32")
            return np.asarray(values, dtype=np.float32), int(decoded_rate)
    return np.asarray(audio, dtype=np.float32), fallback_rate


def eligible(row: dict[str, Any], min_duration: float, max_duration: float) -> bool:
    text = normalize_persian_text(str(row.get("transcript") or ""))
    duration = float(row.get("duration") or 0.0)
    return bool(text) and min_duration <= duration <= max_duration


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    wav_dir = output_dir / "wavs"
    metadata_path = output_dir / "metadata.csv"
    audit_path = output_dir / "metadata.jsonl"

    if metadata_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metadata_path} already exists; pass --overwrite to rebuild it")
    wav_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ManaTTS files: {args.data_files}")
    dataset = load_dataset(
        "MahtaFetrat/Mana-TTS",
        data_files=args.data_files,
        split="train",
    )
    selected: list[dict[str, Any]] = []
    for row in dataset:
        if eligible(row, args.min_duration, args.max_duration):
            selected.append(row)
        if len(selected) >= args.max_samples:
            break
    if not selected:
        raise RuntimeError("No samples passed the transcript and duration filters")

    print(f"Running {args.g2p_model} on {len(selected)} transcripts")
    g2p = PersianG2P(args.g2p_model, device=args.device)
    manifest_lines: list[str] = []
    audit_records: list[dict[str, Any]] = []

    for start in range(0, len(selected), args.batch_size):
        batch = selected[start : start + args.batch_size]
        texts = [normalize_persian_text(str(row["transcript"])) for row in batch]
        phoneme_texts = g2p.phonemize_batch(texts)

        for row, text, phonemes in zip(batch, texts, phoneme_texts, strict=True):
            phonemes = phonemes.replace("|", " ").strip()
            if not phonemes:
                continue
            sample_id = f"sample_{len(manifest_lines):06d}"
            samples, sample_rate = audio_array(row)
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            sf.write(wav_dir / f"{sample_id}.wav", samples, sample_rate, subtype="PCM_16")
            manifest_lines.append(f"{sample_id}|{phonemes}|{phonemes}")
            audit_records.append(
                {
                    "id": sample_id,
                    "source_file": row.get("file_name"),
                    "text": text,
                    "phonemes": phonemes,
                    "duration": float(row.get("duration") or 0.0),
                    "sample_rate": sample_rate,
                    "g2p_model": args.g2p_model,
                }
            )
        print(f"Prepared {len(manifest_lines)}/{len(selected)} samples")

    metadata_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    with audit_path.open("w", encoding="utf-8") as audit_file:
        for record in audit_records:
            audit_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Dataset ready: {output_dir}")
    print(f"Training rows: {len(manifest_lines)}")


if __name__ == "__main__":
    main()
