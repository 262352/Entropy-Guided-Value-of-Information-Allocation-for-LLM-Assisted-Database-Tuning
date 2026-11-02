from __future__ import annotations

from cognitune.evaluator import Evaluator
from cognitune.evaluator.base import EvaluationResult
from cognitune.experiment.recorder import RunRecorder, TrialRecord


DUPLICATE_TRIAL_OFFSET = 100_000
_PROGRESS_TAG = {"value": ""}


def set_progress_tag(value: str | None) -> None:
    _PROGRESS_TAG["value"] = value or ""


def evaluate_and_record(
    evaluator: Evaluator,
    recorder: RunRecorder,
    method: str,
    seed: int,
    trial: int,
    config: dict,
    source: str,
    best_utility: float,
    budget: int | None = None,
    duplicate_new_best: bool = False,
    **extras,
) -> tuple[float, object]:
    best_before = max(
        (record.raw_metric for record in recorder.records if record.status == "SUCCESS"),
        default=None,
    )
    result = evaluator.evaluate(config, trial)
    reps = 1
    if (
        duplicate_new_best
        and result.status == "SUCCESS"
        and best_before is not None
        and result.raw_metric > best_before
    ):
        repeated = evaluator.evaluate(config, trial + DUPLICATE_TRIAL_OFFSET)
        if repeated.status == "SUCCESS":
            result = _average(result, repeated)
            reps = 2
    best_utility = max(best_utility, result.utility)
    recorder.append(TrialRecord(
        trial=trial, method=method, seed=seed, config=config,
        raw_metric=result.raw_metric, utility=result.utility, best_utility=best_utility,
        best_so_far=best_utility, status=result.status, candidate_source=source,
        elapsed_seconds=result.elapsed_seconds, metadata=result.metadata, **extras,
    ))
    _report_progress(method, seed, trial, budget, result, recorder.records, reps)
    return best_utility, result


def _average(first: EvaluationResult, second: EvaluationResult) -> EvaluationResult:
    metadata = dict(first.metadata or {})
    metadata["duplicate_measurement"] = {
        "first_raw_metric": first.raw_metric,
        "second_raw_metric": second.raw_metric,
    }
    return EvaluationResult(
        raw_metric=(first.raw_metric + second.raw_metric) / 2.0,
        utility=(first.utility + second.utility) / 2.0,
        status="SUCCESS",
        elapsed_seconds=first.elapsed_seconds + second.elapsed_seconds,
        restart_seconds=first.restart_seconds + second.restart_seconds,
        warmup_seconds=first.warmup_seconds,
        benchmark_seconds=first.benchmark_seconds + second.benchmark_seconds,
        metadata=metadata,
        config=first.config,
    )


def _report_progress(
    method: str,
    seed: int,
    trial: int,
    budget: int | None,
    result,
    records: list[TrialRecord],
    reps: int = 1,
) -> None:
    successes = [record.raw_metric for record in records if record.status == "SUCCESS"]
    best = f"{max(successes):.3f}" if successes else "n/a"
    total = f"/{budget}" if budget else ""
    repetition = " reps=2" if reps == 2 else ""
    label = f"{_PROGRESS_TAG['value']}/{method}" if _PROGRESS_TAG["value"] else method
    print(
        f"[{label} seed={seed}] round {trial + 1}{total} "
        f"TPS={result.raw_metric:.3f} best_TPS={best} status={result.status}{repetition}",
        flush=True,
    )
