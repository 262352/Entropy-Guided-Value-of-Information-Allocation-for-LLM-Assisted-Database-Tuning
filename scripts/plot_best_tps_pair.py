#!/usr/bin/env python3
"""Plot the observed best-so-far TPS of CogniTune v2 and MCTuner."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parents[1]
METHODS = [("cognitune_v2", "CogniTune (Ours)", "o"),
           ("mctuner_official", "MCTuner", "s")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts/real_pair_50")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="Output file stem (without extension)")
    args = parser.parse_args()
    output = args.output or args.artifacts / "analysis/tpcc_best_tps_current"
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshots = {}
    exported = []
    metadata = {"metric": "raw_metric (logged Goodput)", "series": {}}
    for method, label, marker in METHODS:
        path = args.artifacts / "runs" / method / f"seed_{args.seed}" / "trials.jsonl"
        contents = path.read_text()
        records = []
        for index, line in enumerate(contents.splitlines()):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                if index == len(contents.splitlines()) - 1 and not contents.endswith("\n"):
                    break  # A live writer has not finished its last record yet.
                raise
        if not records:
            raise ValueError(f"No completed trials in {path}")
        trials = [int(r["trial"]) for r in records]
        if trials != sorted(set(trials)):
            raise ValueError(f"Non-unique or unordered trial indices in {path}")
        best, best_round = -math.inf, None
        x, y = [], []
        for row in records:
            round_number = int(row["trial"]) + 1
            raw = float(row["raw_metric"])
            if row["status"] == "SUCCESS" and math.isfinite(raw) and raw > best:
                best, best_round = raw, round_number
            value = best if math.isfinite(best) else math.nan
            x.append(round_number)
            y.append(value)
            exported.append({"method": method, "round": round_number,
                             "status": row["status"], "raw_TPS": raw, "best_TPS": value})
        if best_round is None:
            raise ValueError(f"No successful finite TPS in {path}")
        snapshots[method] = (x, y)
        metadata["series"][method] = {"source": str(path.resolve()), "rounds": len(records),
                                      "last_round": x[-1], "best_TPS": best,
                                      "first_best_round": best_round}
        print(f"{label}: {len(records)} completed rounds; best TPS={best:.3f} at round {best_round}")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 11, "axes.titlesize": 12,
                         "legend.fontsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
    colors = plt.get_cmap("tab10").colors
    for index, (method, label, marker) in enumerate(METHODS):
        x, y = snapshots[method]
        ax.plot(x, y, label=label, color=colors[index], marker=marker,
                markersize=2.8, linewidth=1.5, markevery=1, alpha=0.95)
        ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), xytext=(-5, 9),
                    textcoords="offset points", ha="right", color=colors[index], fontsize=9)
    last_round = max(x[-1] for x, _ in snapshots.values())
    ax.set_xlabel("Tuning round")
    ax.set_ylabel("Best TPS (Goodput)")
    ax.set_title("TPC-C workload: best-so-far TPS")
    ax.set_xlim(1, max(2, last_round))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.20)
    ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    for extension in ("png", "pdf"):
        target = output.with_suffix("." + extension)
        fig.savefig(target, bbox_inches="tight")
        print(target.resolve())
    plt.close(fig)
    with output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
