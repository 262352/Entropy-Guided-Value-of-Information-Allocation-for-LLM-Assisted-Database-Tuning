from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from cognitune.search_space.space import SearchSpace
from .base import EvaluationResult, Evaluator


def _extract_metric(summary_path: str | Path, throughput: float) -> tuple[float, dict]:
    """Return the metric used for tuning plus both reported rates.

    BenchBase's ``Throughput`` counts every measured request, including the
    serialization retries that TPC-C at SERIALIZABLE generates. Those retries
    can inflate throughput by 3x while useful work stays flat. ``Goodput``
    counts completed transactions instead, so it is the primary metric; the
    raw request rate is retained in metadata for diagnosis.
    """
    data = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    goodput = data.get("Goodput (requests/second)")
    if isinstance(goodput, (int, float)) and goodput > 0:
        metric = float(goodput)
        metric_name = "goodput"
    else:
        metric = float(throughput)
        metric_name = "throughput"
    details = {
        "metric": metric_name,
        "throughput_requests_per_second": float(throughput),
        "goodput_requests_per_second": float(goodput) if isinstance(goodput, (int, float)) else None,
        "measured_requests": data.get("Measured Requests"),
    }
    return metric, details


class PostgreSQLBenchBaseEvaluator(Evaluator):
    """Adapter over the already validated GPTuner PostgreSQL/BenchBase harness."""

    def __init__(self, space: SearchSpace, legacy_config: str | Path, workload: str, seed: int, artifact_root: str | Path, n_init: int, failure_penalty: float = 1.0):
        legacy_src = Path("/root/GPTuner/baselines_random/src")
        if str(legacy_src) not in sys.path:
            sys.path.insert(0, str(legacy_src))
        from db_bo_baselines.benchbase import BenchBaseRunner
        from db_bo_baselines.config import load_config
        from db_bo_baselines.postgres import PostgreSQLController

        self.space = space
        self.cfg = load_config(legacy_config)
        self.pg = PostgreSQLController(self.cfg["postgres"])
        self.benchbase = BenchBaseRunner(self.cfg["benchbase"], self.cfg["postgres"])
        self.knobs = self.pg.discover_knobs()
        if set(space.by_name) != {knob.name for knob in self.knobs}:
            raise ValueError("CogniTune and legacy PostgreSQL harness knob sets differ")
        self.workload = workload
        self.workload_cfg = self.cfg["benchbase"]["workloads"][workload]
        self.seed = seed
        self.n_init = n_init
        self.failure_penalty = failure_penalty
        self.artifact_root = Path(artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.shared_initial_dir = self.artifact_root / "shared_initial" / f"seed_{seed}"
        self.shared_initial_dir.mkdir(parents=True, exist_ok=True)
        self.default_metric: float | None = None
        self.minimum_utility = 0.0

    def prepare(self) -> None:
        if not self._ensure_ready():
            raise RuntimeError("PostgreSQL is unavailable and could not be recovered")
        self.pg.assert_version()
        if not self.benchbase.jar.is_file():
            raise FileNotFoundError(f"BenchBase is not built: {self.benchbase.jar}")
        if not self.pg.database_exists(self.workload_cfg["template_database"]):
            raise RuntimeError(f"missing template database {self.workload_cfg['template_database']}")
        self.benchbase.generate_config(self.workload)

    def evaluate(self, config: dict, trial: int) -> EvaluationResult:
        config = self.space.repair(config)
        shared_path = self.shared_initial_dir / f"trial_{trial:03d}.json"
        if trial < self.n_init and shared_path.exists():
            cached = json.loads(shared_path.read_text(encoding="utf-8"))
            if (cached.get("metadata") or {}).get("metric") in {"goodput", "throughput"}:
                result = EvaluationResult(**cached)
                if trial == 0:
                    self.default_metric = result.raw_metric
                self.minimum_utility = min(self.minimum_utility, result.utility)
                result.metadata = (result.metadata or {}) | {"shared_initial_reused": True}
                return result

        started = time.monotonic()
        restart_seconds = 0.0
        try:
            if not self._ensure_ready():
                raise RuntimeError("PostgreSQL is unavailable and could not be recovered")
            self.pg.write_candidate(self.knobs, config)
            phase = time.monotonic()
            self._restart_postgres()
            restart_seconds = time.monotonic() - phase
            self.pg.recreate_database(self.workload_cfg["database"], self.workload_cfg["template_database"])
            effective = self.pg.effective_values(list(config))
            key = hashlib.sha256(self.space.config_key(config).encode()).hexdigest()[:12]
            output = self.artifact_root / "benchbase" / f"seed_{self.seed}" / f"trial_{trial:03d}_{key}"
            phase = time.monotonic()
            throughput, summary = self.benchbase.execute(self.workload, output)
            benchmark_seconds = time.monotonic() - phase
            raw_metric, metric_details = _extract_metric(summary, throughput)
            if self.default_metric is None:
                if trial != 0:
                    raise RuntimeError("default trial must be evaluated before non-default configurations")
                self.default_metric = raw_metric
            utility = math.log(raw_metric / self.default_metric)
            self.minimum_utility = min(self.minimum_utility, utility)
            result = EvaluationResult(
                raw_metric=raw_metric, utility=utility, status="SUCCESS",
                elapsed_seconds=time.monotonic() - started, config=config,
                restart_seconds=restart_seconds,
                warmup_seconds=float(self.workload_cfg.get("warmup_seconds", 0)),
                benchmark_seconds=benchmark_seconds,
                metadata={
                    "effective_config": effective,
                    "benchbase_summary": str(summary),
                    "shared_initial_reused": False,
                    **metric_details,
                },
            )
        except Exception as error:
            result = EvaluationResult(
                raw_metric=0.0, utility=self.minimum_utility - abs(self.failure_penalty), status="CRASH",
                elapsed_seconds=time.monotonic() - started, config=config, restart_seconds=restart_seconds,
                metadata={"error": str(error), "traceback": traceback.format_exc()},
            )
        if trial < self.n_init:
            temporary = shared_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(shared_path)
        return result

    def restore_default(self) -> None:
        try:
            if not self._ensure_ready(attempts=1):
                self._force_stop()
                self.pg.recover()
                self.pg.wait_until_ready()
            self.pg.reset_all()
            self._restart_postgres()
        except Exception as error:
            print(f"[evaluator] restore_default failed: {error}", flush=True)

    def health_check(self) -> bool:
        try:
            self.pg.wait_until_ready()
            return True
        except Exception:
            return False

    def _ensure_ready(self, attempts: int = 2) -> bool:
        """Make the server reachable, with a hard stop/recover fallback."""
        try:
            self.pg.wait_until_ready()
            return True
        except Exception:
            pass
        for _ in range(attempts):
            try:
                self._restart_postgres()
                return True
            except Exception:
                self._force_stop()
                try:
                    self.pg.recover()
                    self.pg.wait_until_ready()
                    return True
                except Exception:
                    time.sleep(2)
        return False

    def _restart_postgres(self) -> None:
        """Restart with checkpoint/recovery-aware timeouts.

        The legacy restart script gives pg_ctl its default 60-second wait.
        Under a 120-second write-heavy benchmark, both fast shutdown's
        checkpoint and crash recovery after an immediate stop can exceed that
        limit, which used to cascade into CRASH rounds. Stop fast with up to
        three minutes for the checkpoint, fall back to immediate/kill, then
        start with up to five minutes for recovery.
        """
        # Immediate shutdown skips the checkpoint.  With a large
        # shared_buffers candidate on a slow overlay disk, the shutdown
        # checkpoint can spend hours flushing dirty pages and leave the
        # checkpointer in uninterruptible I/O (state D).  The benchmark
        # database is recreated from its template after startup, so skipping
        # the checkpoint is safe and far more robust here.
        stopped = self._pg_stop("immediate", 30)
        if not stopped:
            stopped = self._pg_stop("fast", 180)
        if not stopped:
            self._force_stop()
        if not self._pg_start(300):
            # A candidate config that prevents startup must be removed before
            # the next trial; mark this trial as a crash.
            try:
                self.pg.recover()
                self.pg.wait_until_ready()
            except Exception:
                self._force_stop()
                self.pg.recover()
                self.pg.wait_until_ready()
            raise RuntimeError("candidate prevented a clean PostgreSQL restart")
        self.pg.wait_until_ready()

    def _pg_stop(self, mode: str, timeout: int) -> bool:
        pg_ctl, data_dir, owner, log_path, port = self._pg_paths()
        if not (Path(pg_ctl).is_file() and Path(data_dir).is_dir()):
            return False
        command = [pg_ctl, "stop", "-D", data_dir, "-m", mode, "-w", "-t", str(timeout)]
        if os.geteuid() == 0 and owner:
            command = ["runuser", "-u", owner, "--"] + command
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout + 30, cwd="/tmp", check=False)
            return completed.returncode == 0
        except Exception:
            return False

    def _pg_start(self, timeout: int) -> bool:
        pg_ctl, data_dir, owner, log_path, port = self._pg_paths()
        if not (Path(pg_ctl).is_file() and Path(data_dir).is_dir()):
            return False
        command = [
            pg_ctl, "start", "-D", data_dir, "-l", log_path,
            "-w", "-t", str(timeout), "-o", f"-p {port}",
        ]
        if os.geteuid() == 0 and owner:
            command = ["runuser", "-u", owner, "--"] + command
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout + 30, cwd="/tmp", check=False)
            return completed.returncode == 0
        except Exception:
            return False

    def _pg_paths(self):
        return (
            os.environ.get("COGNITUNE_PG_CTL", "/opt/postgresql-14.9/bin/pg_ctl"),
            os.environ.get("COGNITUNE_PG_DATA", "/home/pgbaseline/pgdata-14.9"),
            os.environ.get("COGNITUNE_PG_OWNER", "pgbaseline"),
            os.environ.get("COGNITUNE_PG_LOG", "/home/pgbaseline/postgres-14.9.log"),
            int(os.environ.get("COGNITUNE_PG_PORT", "55432")),
        )

    def _force_stop(self) -> None:
        pg_ctl = os.environ.get("COGNITUNE_PG_CTL", "/opt/postgresql-14.9/bin/pg_ctl")
        data_dir = os.environ.get("COGNITUNE_PG_DATA", "/home/pgbaseline/pgdata-14.9")
        owner = os.environ.get("COGNITUNE_PG_OWNER", "pgbaseline")
        if not (Path(pg_ctl).is_file() and Path(data_dir).is_dir()):
            return
        command = [pg_ctl, "stop", "-D", data_dir, "-m", "immediate", "-w", "-t", "15"]
        if os.geteuid() == 0 and owner:
            command = ["runuser", "-u", owner, "--"] + command
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=60, cwd="/tmp", check=False)
            if completed.returncode == 0:
                return
            # A backend in uninterruptible I/O can keep even an immediate
            # shutdown waiting. This host runs a disposable benchmark
            # database, so killing the cluster and letting PostgreSQL recover
            # from WAL is preferable to a 25-minute stall.
            kill = ["pkill", "-9", "-u", owner, "-f", "/opt/postgresql-14.9/bin/postgres"]
            subprocess.run(kill, capture_output=True, text=True, timeout=30, cwd="/tmp", check=False)
            time.sleep(1)
            pid_file = Path(data_dir) / "postmaster.pid"
            if pid_file.exists():
                try:
                    pid = int(pid_file.read_text(encoding="utf-8").splitlines()[0])
                    os.kill(pid, 0)
                except Exception:
                    pid_file.unlink(missing_ok=True)
        except Exception:
            pass
