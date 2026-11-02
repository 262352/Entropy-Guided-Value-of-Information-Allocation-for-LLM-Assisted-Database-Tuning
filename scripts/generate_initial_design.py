#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from cognitune.config import load_search_space
from cognitune.search_space.initial_design import build_initial_design, save_initial_design


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True)
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output")
    args = parser.parse_args()
    space = load_search_space(args.space)
    output = Path(args.output or f"artifacts/initial_design/{Path(args.space).stem}_seed_{args.seed}.json")
    save_initial_design(output, build_initial_design(space, args.seed, args.n_init))
    print(output)


if __name__ == "__main__":
    main()
