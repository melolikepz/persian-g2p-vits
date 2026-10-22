#!/usr/bin/env python3
"""Generate a WAV sample from a base or fine-tuned Persian MMS/VITS model."""

import argparse
from pathlib import Path

import scipy.io.wavfile
import torch
from transformers import AutoTokenizer, VitsModel


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/mms-tts-fas")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/sample.wav"))
    args = parser.parse_args()

    device = select_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = VitsModel.from_pretrained(args.model).to(device)
    model.eval()

    inputs = tokenizer(args.text, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        waveform = model(**inputs).waveform[0].float().cpu().numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.wavfile.write(args.output, model.config.sampling_rate, waveform)
    print(f"Saved {args.output} ({model.config.sampling_rate} Hz, device={device})")


if __name__ == "__main__":
    main()
