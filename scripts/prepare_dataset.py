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
        streaming=True,
    )
    print(f"Running {args.g2p_model} on up to {args.max_samples} transcripts")
    g2p = PersianG2P(args.g2p_model, device=args.device)

    selected_count = 0
    prepared_count = 0
    batch: list[dict[str, Any]] = []

    def prepare_batch(
        rows: list[dict[str, Any]], manifest_file: Any, audit_file: Any
    ) -> int:
        nonlocal prepared_count
        if not rows:
            return 0

        texts = [normalize_persian_text(str(row["transcript"])) for row in rows]
        phoneme_texts = g2p.phonemize_batch(texts)

        written = 0
        for row, text, phonemes in zip(rows, texts, phoneme_texts, strict=True):
            phonemes = phonemes.replace("|", " ").strip()
            if not phonemes:
                continue
            sample_id = f"sample_{prepared_count:06d}"
            samples, sample_rate = audio_array(row)
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            sf.write(wav_dir / f"{sample_id}.wav", samples, sample_rate, subtype="PCM_16")
            manifest_file.write(f"{sample_id}|{phonemes}|{phonemes}\n")
            audit_file.write(
                json.dumps(
                    {
                        "id": sample_id,
                        "source_file": row.get("file_name"),
                        "text": text,
                        "phonemes": phonemes,
                        "duration": float(row.get("duration") or 0.0),
                        "sample_rate": sample_rate,
                        "g2p_model": args.g2p_model,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            prepared_count += 1
            written += 1

        manifest_file.flush()
        audit_file.flush()
        print(f"Prepared {prepared_count}/{args.max_samples} samples", flush=True)
        return written

    with (
        metadata_path.open("w", encoding="utf-8") as manifest_file,
        audit_path.open("w", encoding="utf-8") as audit_file,
    ):
        for row in dataset:
            if not eligible(row, args.min_duration, args.max_duration):
                continue
            batch.append(row)
            selected_count += 1

            if len(batch) >= args.batch_size:
                prepare_batch(batch, manifest_file, audit_file)
                batch.clear()

            if selected_count >= args.max_samples:
                break

        prepare_batch(batch, manifest_file, audit_file)

    if prepared_count == 0:
        raise RuntimeError("No samples passed the transcript and duration filters")

    print(f"Dataset ready: {output_dir}")
    print(f"Training rows: {prepared_count}")


if __name__ == "__main__":
    main()
