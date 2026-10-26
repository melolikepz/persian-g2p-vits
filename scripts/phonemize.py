#!/usr/bin/env python3
"""Print the Homo-GE2PE phoneme sequence for one Persian sentence."""

from __future__ import annotations

import argparse

from persian_g2p_vits import DEFAULT_G2P_MODEL, PersianG2P


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--g2p-model", default=DEFAULT_G2P_MODEL)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    args = parser.parse_args()
    print(PersianG2P(args.g2p_model, device=args.device).phonemize(args.text))


if __name__ == "__main__":
    main()
