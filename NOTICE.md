# Third-party notices

This repository orchestrates third-party software, model weights, and data. They are not relicensed by this repository.

- **Homo-GE2PE Persian**: the default checkpoint is the Transformers-compatible `mehdi-hf/Homo-GE2PE-Persian-HF` packaging. Its model card reports byte-identical weights to `MahtaFetrat/Homo-GE2PE-Persian` and an MIT license. Cite the original GE2PE and Homo-GE2PE authors.
- **Coqui AI TTS**: installed as the `coqui-tts` dependency and licensed under MPL-2.0.
- **ManaTTS**: downloaded at runtime from Hugging Face and released under CC0-1.0.
- **ByT5**: the G2P checkpoint is based on ByT5, whose upstream model and code use the Apache-2.0 license.

Review the upstream model cards and repositories before publishing or distributing trained checkpoints.
