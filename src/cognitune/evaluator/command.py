from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from .base import EvaluationResult, Evaluator, metric_utility


class CommandEvaluator(Evaluator):
    def __init__(
        self,
        command: list[str],
        objective: str,
        default_metric: float,
        timeout_seconds: int = 900,
        metric_regex: str | None = None,
        result_json: str | None = None,
        apply_command: list[str] | None = None,
        restart_command: list[str] | None = None,
        warmup_command: list[str] | None = None,
        reset_command: list[str] | None = None,
        health_command: list[str] | None = None,
        restore_command: list[str] | None = None,
        failure_penalty: float = 1.0,
        workdir: str | None = None,
    ):
        self.command = command
        self.objective = objective
        self.default_metric = default_metric
        self.timeout_seconds = timeout_seconds
        self.metric_regex = re.compile(metric_regex) if metric_regex else None
        self.result_json = result_json
        self.apply_command = apply_command
        self.restart_command = restart_command
        self.warmup_command = warmup_command
        self.reset_command = reset_command
        self.health_command = health_command
        self.restore_command = restore_command
        self.failure_penalty = failure_penalty
        self.workdir = workdir
        self.minimum_observed_utility: float | None = None

    def evaluate(self, config: dict, trial: int) -> EvaluationResult:
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="cognitune-") as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            try:
                self._run_optional(self.reset_command, config_path, trial)
                self._run_optional(self.apply_command, config_path, trial)
                phase = time.monotonic()
                self._run_optional(self.restart_command, config_path, trial)
                restart_seconds = time.monotonic() - phase
                self._run_optional(self.health_command, config_path, trial)
                phase = time.monotonic()
                self._run_optional(self.warmup_command, config_path, trial)
                warmup_seconds = time.monotonic() - phase
                phase = time.monotonic()
                process = self._run(self.command, config_path, trial)
                benchmark_seconds = time.monotonic() - phase
                metric = self._extract_metric(process.stdout, config_path, trial)
                utility = metric_utility(metric, self.default_metric, self.objective)
                self.minimum_observed_utility = utility if self.minimum_observed_utility is None else min(self.minimum_observed_utility, utility)
                return EvaluationResult(
                    metric, utility, "SUCCESS", elapsed_seconds=time.monotonic() - started,
                    metadata={"stdout": process.stdout[-4000:], "stderr": process.stderr[-4000:]}, config=config,
                    restart_seconds=restart_seconds, warmup_seconds=warmup_seconds,
                    benchmark_seconds=benchmark_seconds,
                )
            except subprocess.TimeoutExpired as error:
                return self._failure(config, started, "TIMEOUT", error)
            except (subprocess.SubprocessError, OSError, ValueError, KeyError) as error:
                return self._failure(config, started, "CRASH", error)
            finally:
                try:
                    self._run_optional(self.restore_command, config_path, trial)
                except (subprocess.SubprocessError, OSError):
                    pass

    def _failure(self, config, started, status, error):
        floor = self.minimum_observed_utility if self.minimum_observed_utility is not None else 0.0
        return EvaluationResult(
                    0.0, floor - abs(self.failure_penalty), status, time.monotonic() - started,
                    {"error": str(error)}, config=config,
                )

    def _run(self, command: list[str], config_path: Path, trial: int):
        rendered = [part.format(config_json=str(config_path), trial=trial) for part in command]
        return subprocess.run(
            rendered, cwd=self.workdir, text=True, capture_output=True,
            timeout=self.timeout_seconds, check=True,
        )

    def _run_optional(self, command, config_path, trial):
        if command:
            self._run(command, config_path, trial)

    def _extract_metric(self, stdout: str, config_path: Path, trial: int) -> float:
        if self.result_json:
            path = Path(self.result_json.format(config_json=str(config_path), trial=trial))
            return float(json.loads(path.read_text(encoding="utf-8"))["metric"])
        if not self.metric_regex:
            raise ValueError("metric_regex or result_json is required")
        match = self.metric_regex.search(stdout)
        if not match:
            raise ValueError("metric pattern did not match command output")
        return float(match.group("metric") if "metric" in match.groupdict() else match.group(1))

    def reset(self) -> None:
        if self.reset_command:
            self._run(self.reset_command, Path("/dev/null"), -1)

    def restore_default(self) -> None:
        if self.restore_command:
            self._run(self.restore_command, Path("/dev/null"), -1)

    def health_check(self) -> bool:
        if not self.health_command:
            return True
        try:
            self._run(self.health_command, Path("/dev/null"), -1)
            return True
        except (subprocess.SubprocessError, OSError):
            return False
