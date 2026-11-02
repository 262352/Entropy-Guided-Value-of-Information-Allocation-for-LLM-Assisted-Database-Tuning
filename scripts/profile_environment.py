#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cognitune.environment import collect_environment_profile, write_postgres_search_space


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchbase-xml", default="/root/GPTuner/baselines_random/generated_configs/tpcc.xml")
    parser.add_argument("--manifest", default="/root/GPTuner/baselines_random/configs/postgres_performance_space.json")
    parser.add_argument("--profile-output", default="artifacts/profiles/postgresql_tpcc.json")
    parser.add_argument("--space-output", default="configs/knobs/postgresql_60_live.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=55432)
    parser.add_argument("--user", default="benchadmin")
    args = parser.parse_args()
    profile = collect_environment_profile(args.benchbase_xml, args.host, args.port, args.user)
    destination = Path(args.profile_output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_postgres_search_space(profile, args.manifest, args.space_output)
    print(destination)


if __name__ == "__main__":
    main()
