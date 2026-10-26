"""Homo-GE2PE inference helpers used by training and synthesis."""

from __future__ import annotations

import re
from collections.abc import Iterable

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

DEFAULT_G2P_MODEL = "mehdi-hf/Homo-GE2PE-Persian-HF"

# Homo-GE2PE's compact output alphabet -> the notation used by the authors.
_TO_REFERENCE_NOTATION = str.maketrans(
    {
        "/": "a",
        "a": "A",
        "@": "?",
        "$": "S",
        "c": "C",
    }
)


def default_device() -> str:
    """Choose the fastest available PyTorch device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_persian_text(text: str) -> str:
    """Apply conservative Unicode and whitespace normalization for Persian."""
    replacements = {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ۀ": "هٔ",
        "ة": "ه",
        "ـ": "",
        "\u200f": "",
        "\u200e": "",
    }
    normalized = text
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    normalized = normalized.replace("\u200c", " ")
    return re.sub(r"\s+", " ", normalized).strip()


def convert_model_output(raw: str) -> str:
    """Convert Homo-GE2PE's internal symbols to its documented notation."""
    converted = raw.translate(_TO_REFERENCE_NOTATION).replace("1", "")
    return re.sub(r"\s+", " ", converted).strip()


class PersianG2P:
    """Batch grapheme-to-phoneme inference with Homo-GE2PE."""

    def __init__(self, model_id: str = DEFAULT_G2P_MODEL, device: str | None = None):
        self.model_id = model_id
        self.device = device or default_device()
        # The model card stores `extra_special_tokens` in the legacy list form.
        # Transformers 4.57+ expects a mapping, so override that one field while
        # preserving every actual tokenizer setting from the checkpoint.
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, extra_special_tokens={})
        self.model = T5ForConditionalGeneration.from_pretrained(model_id)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def phonemize_batch(
        self,
        texts: Iterable[str],
        *,
        num_beams: int = 5,
        max_length: int = 512,
    ) -> list[str]:
        normalized = [normalize_persian_text(text) for text in texts]
        if not normalized:
            return []

        encoded = self.tokenizer(
            normalized,
            padding=True,
            add_special_tokens=False,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)
        generated = self.model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            num_beams=num_beams,
            min_length=1,
            max_length=max_length,
            early_stopping=True,
        )
        raw_outputs = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [convert_model_output(output.strip()) for output in raw_outputs]

    def phonemize(self, text: str, **generation_kwargs: int) -> str:
        """Convert one Persian sentence to a phoneme string."""
        return self.phonemize_batch([text], **generation_kwargs)[0]
