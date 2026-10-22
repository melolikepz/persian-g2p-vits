# Persian MMS/VITS Fine-Tuning

Fine-tune Meta's Persian [`facebook/mms-tts-fas`](https://huggingface.co/facebook/mms-tts-fas)
checkpoint on the Persian [ManaTTS](https://huggingface.co/datasets/MahtaFetrat/Mana-TTS)
dataset.

The base checkpoint uses the VITS architecture and is part of Meta's Massively
Multilingual Speech (MMS) project. This repository adapts the training workflow
from [`ylacombe/finetune-hf-vits`](https://github.com/ylacombe/finetune-hf-vits)
for Persian text-to-speech.

> **License notice:** MMS model weights are licensed under CC BY-NC 4.0.
> Fine-tuned weights derived from them are intended for non-commercial use
> unless additional permission is granted by Meta.

## Features

- Fine-tunes a Persian MMS/VITS checkpoint instead of training from scratch.
- Uses a 1,000-sample subset of ManaTTS by default.
- Includes automatic preparation of the VITS discriminator checkpoint.
- Saves model, tokenizer, and feature-extractor files in one output directory.
- Provides a command-line inference script for generating WAV files.
- Automatically resumes from the latest saved checkpoint.

## Requirements

- Python 3.10 or 3.11
- Git
- An internet connection for downloading the model and dataset
- A supported GPU is recommended; CPU training is possible but slow
- Approximately 15 GB of free disk space

## Quick Start

Clone the repository and enter its directory:

```bash
git clone https://github.com/melolikepz/persian-mms-tts-finetuning.git
cd persian-mms-tts-finetuning
```

Create the Python environment, install dependencies, compile the Cython
alignment module, and prepare the Persian training checkpoint:

```bash
./scripts/setup_local.sh
```

Start fine-tuning:

```bash
./scripts/train_local.sh
```

The trained model and intermediate checkpoints are written to:

```text
outputs/persian-mms-tts
```

## Generate Speech

Run inference with a custom Persian sentence:

```bash
./scripts/infer_local.sh "YOUR_PERSIAN_TEXT"
```

The generated audio is saved as:

```text
outputs/sample.wav
```

If a fine-tuned checkpoint is not available yet, the inference script uses the
base `facebook/mms-tts-fas` model.

## Manual Workflow

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Prepare the trainable Persian checkpoint with its discriminator:

```bash
python convert_original_discriminator_checkpoint.py \
  --language_code fas \
  --pytorch_dump_folder_path checkpoints/mms-tts-fas-train
```

Start training:

```bash
accelerate launch \
  --num_processes 1 \
  --mixed_precision no \
  run_vits_finetuning.py \
  training_config_examples/finetune_mms_fa_local.json
```

Generate a WAV file directly with Python:

```bash
python scripts/infer.py \
  --model outputs/persian-mms-tts \
  --text "YOUR_PERSIAN_TEXT" \
  --output outputs/sample.wav
```

## Training Configuration

The default Persian configuration is located at
[`training_config_examples/finetune_mms_fa_local.json`](training_config_examples/finetune_mms_fa_local.json).

Important settings include:

| Setting | Default | Description |
|---|---:|---|
| `train_split_name` | `train[25:1025]` | Uses 1,000 training samples |
| `eval_split_name` | `train[:25]` | Uses 25 validation samples |
| `num_train_epochs` | `20` | Number of training epochs |
| `per_device_train_batch_size` | `2` | Training batch size per device |
| `gradient_accumulation_steps` | `4` | Effective batch-size multiplier |
| `learning_rate` | `0.00002` | Generator and discriminator learning rate |
| `save_steps` | `250` | Checkpoint save interval |

For a short test run, reduce the training range and the number of epochs. If a
run is interrupted, execute the training command again to resume from the most
recent checkpoint.

## Project Structure

```text
.
├── checkpoints/                 # Prepared base model; excluded from Git
├── outputs/                     # Fine-tuned models and generated audio
├── scripts/
│   ├── setup_local.sh           # Environment and checkpoint preparation
│   ├── train_local.sh           # Fine-tuning entry point
│   ├── infer_local.sh           # Inference helper
│   └── infer.py                 # Python inference script
├── training_config_examples/
│   └── finetune_mms_fa_local.json
├── convert_original_discriminator_checkpoint.py
└── run_vits_finetuning.py
```

## Scientific Background

This repository is an implementation workflow rather than a standalone research
paper. It combines three published components:

1. **VITS architecture** — *Conditional Variational Autoencoder with Adversarial
   Learning for End-to-End Text-to-Speech*, ICML 2021.
2. **MMS base model** — *Scaling Speech Technology to 1,000+ Languages*, JMLR
   2024.
3. **ManaTTS dataset** — *ManaTTS Persian: a Recipe for Creating TTS Datasets
   for Lower Resource Languages*, NAACL 2025.

Full publication links and citation details are available in
[`REFERENCES.md`](REFERENCES.md) and [`CITATION.cff`](CITATION.cff).

## Acknowledgements

The training implementation is derived from Younes Lacombe's MIT-licensed
[`finetune-hf-vits`](https://github.com/ylacombe/finetune-hf-vits) project. See
[`LICENSE`](LICENSE) for the source-code license. The source-code license does
not replace or override the license of the Meta MMS model weights.
