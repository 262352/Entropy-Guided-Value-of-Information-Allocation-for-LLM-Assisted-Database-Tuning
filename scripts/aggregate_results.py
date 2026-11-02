#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from cognitune.experiment.aggregate import aggregate_runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate_runs(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
