"""Small compatibility bridge for Coqui TTS with Transformers 5.x."""

from __future__ import annotations


def prepare_coqui_transformers_compatibility() -> None:
    """Restore the helper imported by Coqui 0.27.x and removed in Transformers 5."""
    import torch
    import transformers.pytorch_utils as transformers_torch_utils

    if not hasattr(transformers_torch_utils, "isin_mps_friendly"):

        def isin_mps_friendly(elements: torch.Tensor, test_elements: torch.Tensor) -> torch.Tensor:
            return torch.isin(elements, test_elements)

        transformers_torch_utils.isin_mps_friendly = isin_mps_friendly
