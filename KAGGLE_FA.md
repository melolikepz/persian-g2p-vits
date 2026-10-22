# Fine-tune Persian MMS/VITS on Kaggle

This repository is a small Kaggle-oriented adaptation of
[`ylacombe/finetune-hf-vits`](https://github.com/ylacombe/finetune-hf-vits).
The model starts from Meta's Persian checkpoint
[`facebook/mms-tts-fas`](https://huggingface.co/facebook/mms-tts-fas) and fine-tunes
on a shard of the open [ManaTTS Persian dataset](https://huggingface.co/datasets/MahtaFetrat/Mana-TTS).

## Run

1. Create a Kaggle notebook.
2. In **Notebook options**, enable a GPU and Internet.
3. Upload `notebooks/persian_mms_tts_kaggle.ipynb` with **File > Import Notebook**.
4. Run all cells from top to bottom.
5. Download `persian-mms-tts.zip` from `/kaggle/working` when training finishes.

The notebook is self-contained: it clones the pinned upstream repository and applies the
small Kaggle compatibility changes itself. The default run uses 1,000 training examples and 20 epochs so the complete pipeline can
be checked within a Kaggle session. Increase `TRAIN_SAMPLES` and `EPOCHS` in the settings
cell for a serious run. A full-dataset run may require several Kaggle sessions; checkpoints
are kept in the output directory and the upstream script supports resuming.

## Licenses

- Upstream fine-tuning code: MIT.
- ManaTTS dataset: CC0-1.0.
- Meta MMS model weights: CC BY-NC 4.0 (non-commercial).

Fine-tuned weights inherit the restrictions of the base MMS checkpoint. Do not use these
weights commercially without choosing a base model whose license permits that use.
