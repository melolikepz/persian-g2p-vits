# Persian G2P VITS

Train a single-speaker Persian text-to-speech model with an explicit, mandatory grapheme-to-phoneme stage.

```text
Persian text -> Homo-GE2PE -> phoneme sequence -> Coqui VITS -> waveform
```

This is a train-from-scratch VITS recipe, not an MMS checkpoint fine-tune. During dataset preparation, every ManaTTS transcript is converted to phonemes by Homo-GE2PE. VITS is trained only on those phoneme strings. During synthesis, Persian text must therefore pass through the same G2P model before it can be given to VITS.

## Components

- **G2P:** [`mehdi-hf/Homo-GE2PE-Persian-HF`](https://huggingface.co/mehdi-hf/Homo-GE2PE-Persian-HF), a Transformers-compatible packaging of the original [`MahtaFetrat/Homo-GE2PE-Persian`](https://huggingface.co/MahtaFetrat/Homo-GE2PE-Persian) weights.
- **Acoustic model and vocoder:** [Coqui VITS](https://github.com/idiap/coqui-ai-TTS).
- **Training data:** [ManaTTS](https://huggingface.co/datasets/MahtaFetrat/Mana-TTS), a 114-hour single-speaker Persian corpus.

## Requirements

- Python 3.10 to 3.13
- Git
- At least 15 GB of free disk space for a practical experiment
- A CUDA-capable GPU is strongly recommended for full training; CPU training is possible but very slow

## Quick start

Clone the repository and create the environment:

```bash
git clone https://github.com/melolikepz/persian-g2p-vits.git
cd persian-g2p-vits
bash scripts/setup.sh
```

Prepare 1,000 utterances from the first ManaTTS shard. This downloads the dataset and G2P model, writes audio files, and stores the generated phonemes in an LJSpeech-style manifest:

```bash
bash scripts/prepare.sh --max-samples 1000
```

Start VITS training:

```bash
bash scripts/train.sh --epochs 1000 --batch-size 16
```

The checkpoints and TensorBoard logs are written below `outputs/`. To monitor a run:

```bash
source .venv/bin/activate
tensorboard --logdir outputs
```

Synthesize Persian text with the newest trained checkpoint:

```bash
source .venv/bin/activate
python scripts/infer.py \
  --text "سلام، این یک آزمایش تبدیل متن فارسی به گفتار است." \
  --output outputs/example.wav
```

`scripts/infer.py` first runs Homo-GE2PE, prints the generated phoneme sequence, and then passes that sequence to VITS. It can also use explicit model files:

```bash
python scripts/infer.py \
  --text "امروز هوا بسیار خوب است." \
  --checkpoint outputs/<run>/best_model.pth \
  --config outputs/<run>/config.json \
  --output outputs/weather.wav
```

## Notebook

The complete workflow is also available in [`notebooks/persian_g2p_vits_training.ipynb`](notebooks/persian_g2p_vits_training.ipynb). Select the **Persian G2P VITS** kernel created by `scripts/setup.sh`, then run the cells in order.

The notebook covers:

1. environment verification;
2. Persian G2P preview;
3. ManaTTS preparation;
4. VITS training;
5. synthesis and audio playback.

## Dataset preparation details

The default command reads `dataset/dataset_part_001.parquet` from ManaTTS and keeps utterances between 1 and 15 seconds. Increase the sample count or select multiple Parquet files for a serious run:

```bash
bash scripts/prepare.sh \
  --data-files "dataset/dataset_part_*.parquet" \
  --max-samples 50000 \
  --batch-size 32
```

Generated files:

```text
data/manatts-g2p/
├── metadata.csv       # id|phonemes|phonemes, consumed by Coqui
├── metadata.jsonl     # original text, phonemes, and source metadata for auditing
└── wavs/              # training waveforms
```

The generated data is intentionally excluded from Git.

## Why `use_phonemes=False` appears in the VITS config

Coqui's `use_phonemes` option means “ask Coqui to run its own phonemizer.” This project performs phonemization explicitly with Homo-GE2PE before Coqui sees the text. The VITS manifest already contains phonemes, so `use_phonemes=False` prevents accidental second phonemization. The model input is still phonemic, and Persian graphemes cannot be used directly with the trained VITS checkpoint.

## Useful options

Use a different device for G2P:

```bash
python scripts/prepare_dataset.py --device cuda
python scripts/infer.py --device cuda --text "سلام"
```

Run a small smoke test:

```bash
bash scripts/prepare.sh --max-samples 50
bash scripts/train.sh --epochs 2 --batch-size 4 --eval-batch-size 2
```

Resume a Coqui run from its run directory:

```bash
source .venv/bin/activate
python -m TTS.bin.train_tts --continue_path outputs/<run-directory>
```

## Repository layout

```text
src/persian_g2p_vits/g2p.py       Homo-GE2PE wrapper and normalization
scripts/prepare_dataset.py         ManaTTS download and phoneme manifest creation
scripts/train_vits.py              Coqui VITS training recipe
scripts/infer.py                   End-to-end Persian synthesis
notebooks/                         Executable training notebook
```

## License and attribution

The original code in this repository is MIT licensed. External models, datasets, and libraries retain their own licenses:

- Homo-GE2PE Persian model packaging: MIT; credit the original GE2PE/Homo-GE2PE authors.
- Coqui AI TTS: MPL-2.0.
- ManaTTS: CC0-1.0.

See [`NOTICE.md`](NOTICE.md) and [`REFERENCES.md`](REFERENCES.md) before redistributing trained artifacts.
