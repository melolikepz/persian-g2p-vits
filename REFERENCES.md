# Papers and sources

## Architecture: VITS

Jaehyeon Kim, Jungil Kong, and Juhee Son. “Conditional Variational Autoencoder
with Adversarial Learning for End-to-End Text-to-Speech.” ICML 2021.

- Paper: https://proceedings.mlr.press/v139/kim21f/kim21f.pdf
- arXiv: https://arxiv.org/abs/2106.06103

## Base multilingual model: MMS

Vineel Pratap et al. “Scaling Speech Technology to 1,000+ Languages.” JMLR 25,
2024.

- Paper: https://jmlr.org/papers/volume25/23-1318/23-1318.pdf
- arXiv: https://arxiv.org/abs/2305.13516
- Persian checkpoint: https://huggingface.co/facebook/mms-tts-fas

## Persian dataset: ManaTTS

Mahta Fetrat Qharabagh, Zahra Dehghanian, and Hamid R. Rabiee. “ManaTTS
Persian: a recipe for creating TTS datasets for lower resource languages.”
NAACL 2025.

- Paper: https://aclanthology.org/2025.naacl-long.464/
- Dataset: https://huggingface.co/datasets/MahtaFetrat/Mana-TTS

## Training implementation

This repository is derived from Younes Lacombe's MIT-licensed
`finetune-hf-vits` implementation:

- https://github.com/ylacombe/finetune-hf-vits
