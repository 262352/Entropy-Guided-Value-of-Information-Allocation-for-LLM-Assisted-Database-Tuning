from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import bootstrap_mean_interval


def plot_best_curves(curves: dict[str, list[list[float]]], output: str | Path) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4.2))
    colors = {"cognitune": "#167D73", "smac": "#C44E52", "gp_bo": "#4C72B0", "cognitune_no_relation": "#DD8452"}
    for method, items in sorted(curves.items()):
        matrix = np.asarray(items, dtype=float)
        mean, lower, upper = bootstrap_mean_interval(matrix)
        trials = np.arange(1, matrix.shape[1] + 1)
        color = colors.get(method)
        axis.plot(trials, mean, label=method, color=color)
        axis.fill_between(trials, lower, upper, color=color, alpha=0.16)
    axis.set_xlabel("Database evaluations")
    axis.set_ylabel("Best-so-far utility")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination


def plot_individual_curves(curves: dict[str, list[list[float]]], output: str | Path) -> Path:
    destination = Path(output)
    figure, axis = plt.subplots(figsize=(7, 4.2))
    for method, items in sorted(curves.items()):
        for seed, values in enumerate(items):
            axis.plot(range(1, len(values) + 1), values, alpha=0.65, label=f"{method} seed {seed}")
    axis.set_xlabel("Database evaluations")
    axis.set_ylabel("Best-so-far utility")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination


def plot_cognitive_diagnostics(records: list[dict], output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generated = []
    for field, filename, ylabel in (
        ("selection_entropy", "selection_entropy.png", "Normalized selection entropy"),
        ("value_entropy", "value_entropy.png", "Value recommendation disagreement"),
    ):
        values = records[0].get(field) or {}
        figure, axis = plt.subplots(figsize=(7, 4.2))
        axis.bar(range(len(values)), list(values.values()), color="#4C72B0")
        axis.set_xticks(range(len(values)), list(values), rotation=35, ha="right")
        axis.set_ylabel(ylabel)
        figure.tight_layout()
        destination = output / filename
        figure.savefig(destination, dpi=160)
        plt.close(figure)
        generated.append(destination)
    trials = [record["trial"] + 1 for record in records]
    active = [sum(item.get("state") == "ACTIVE" for item in (record.get("knob_states") or {}).values()) for record in records]
    blocks = [len(record.get("coordination_blocks") or []) for record in records]
    for values, filename, ylabel in (
        (active, "active_knobs.png", "Active knobs"),
        (blocks, "coordination_blocks.png", "Coordination blocks"),
    ):
        figure, axis = plt.subplots(figsize=(7, 4.2))
        axis.step(trials, values, where="post", color="#167D73")
        axis.set_xlabel("Database evaluations")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.22)
        figure.tight_layout()
        destination = output / filename
        figure.savefig(destination, dpi=160)
        plt.close(figure)
        generated.append(destination)
    return generated
