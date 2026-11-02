from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse
from pathlib import Path
from typing import Any

import psycopg2
import yaml


TRANSACTION_MODES = {
    "NewOrder": "read_write",
    "Payment": "read_write",
    "OrderStatus": "read_only",
    "Delivery": "read_write",
    "StockLevel": "read_only",
}


def collect_environment_profile(
    benchbase_xml: str | Path,
    host: str = "127.0.0.1",
    port: int = 55432,
    user: str = "benchadmin",
    password_env: str = "PGPASSWORD",
) -> dict[str, Any]:
    password = os.environ.get(password_env)
    if not password:
        raise ValueError(f"{password_env} is required to inspect PostgreSQL")
    workload = _workload_profile(Path(benchbase_xml))
    postgres = _postgres_profile(host, port, user, password, workload["database"])
    return {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "hardware": _hardware_profile(),
        "dbms": postgres,
        "workload": workload,
        "evidence_notes": {
            "transaction_mix": "parsed from the exact generated BenchBase XML",
            "transaction_read_write_ratio": "classification by transaction semantics, not statement count",
            "database_counters": "cumulative PostgreSQL counters; not attributed to one benchmark interval",
            "hard_constraints": "derived conservatively from host resources and workload concurrency",
        },
    }


def write_postgres_search_space(profile: dict[str, Any], manifest_path: str | Path, destination: str | Path, password_env: str = "PGPASSWORD") -> Path:
    password = os.environ.get(password_env)
    if not password:
        raise ValueError(f"{password_env} is required to inspect PostgreSQL")
    dbms = profile["dbms"]
    with psycopg2.connect(host=dbms["host"], port=dbms["port"], database="postgres", user=dbms["user"], password=password) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT name, unit, context, vartype, reset_val, category, short_desc FROM pg_settings")
            settings = {row[0]: row for row in cursor.fetchall()}
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    knobs = []
    for declared in manifest["knobs"]:
        name = declared["name"]
        row = settings[name]
        _, unit, context, vartype, reset, category, description = row
        item: dict[str, Any] = {
            "name": name,
            "type": {"real": "float", "enum": "categorical"}.get(vartype, vartype),
            "default": _typed_default(vartype, reset),
            "category": _category(category, name),
            "restart_required": context == "postmaster",
            "description": description,
        }
        if unit:
            item["unit"] = unit
        if vartype in {"integer", "real"}:
            item["lower"] = int(declared["min"]) if vartype == "integer" else float(declared["min"])
            item["upper"] = int(declared["max"]) if vartype == "integer" else float(declared["max"])
            item["log"] = item["lower"] > 0 and item["upper"] / item["lower"] >= 100
        elif vartype == "enum":
            item["choices"] = declared.get("choices") or []
        knobs.append(item)
    memory_kib = int(profile["hardware"]["memory_bytes"] * 0.80 / 1024)
    terminals = int(profile["workload"]["terminals"])
    document = {
        "profile_source": str(Path(manifest_path).resolve()),
        "knobs": knobs,
        "hard_constraints": [{
            "type": "sum_le", "limit": memory_kib,
            "terms": [
                {"knob": "shared_buffers", "multiplier": 8},
                {"knob": "work_mem", "multiplier": max(2, terminals * 2)},
                {"knob": "maintenance_work_mem", "multiplier": 1},
                {"knob": "temp_buffers", "multiplier": terminals},
            ],
            "description": "Conservative 80% host-memory envelope with two work_mem operators per terminal",
        }],
    }
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return output


def _hardware_profile() -> dict[str, Any]:
    lscpu = _parse_key_values(_run(["lscpu"]))
    meminfo = _parse_meminfo(Path("/proc/meminfo").read_text())
    disks = []
    raw = json.loads(_run(["lsblk", "-J", "-b", "-o", "NAME,TYPE,SIZE,ROTA,MODEL,MOUNTPOINTS"]))
    for item in raw.get("blockdevices", []):
        if item.get("type") == "disk":
            disks.append({key: item.get(key) for key in ("name", "size", "rota", "model", "mountpoints")})
    return {
        "hostname": platform.node(), "kernel": platform.release(), "architecture": platform.machine(),
        "cpu_model": lscpu.get("Model name"), "logical_cpus": int(lscpu.get("CPU(s)", 0)),
        "cores_per_socket": int(lscpu.get("Core(s) per socket", 0)), "sockets": int(lscpu.get("Socket(s)", 0)),
        "threads_per_core": int(lscpu.get("Thread(s) per core", 0)), "numa_nodes": int(lscpu.get("NUMA node(s)", 0)),
        "l3_cache": lscpu.get("L3 cache"), "memory_bytes": meminfo["MemTotal"] * 1024,
        "memory_available_bytes": meminfo["MemAvailable"] * 1024, "swap_bytes": meminfo.get("SwapTotal", 0) * 1024,
        "cgroup_cpu_max": _read_optional("/sys/fs/cgroup/cpu.max"),
        "cgroup_memory_max": _read_optional("/sys/fs/cgroup/memory.max"),
        "disks": disks,
    }


def _workload_profile(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    names = [node.findtext("name") for node in root.findall(".//transactiontype")]
    weights = [float(value) for value in root.findtext(".//works/work/weights", "").split(",")]
    mix = []
    for name, weight in zip(names, weights):
        mix.append({"transaction": name, "weight_percent": weight, "access_mode": TRANSACTION_MODES.get(name, "unknown")})
    read_write = sum(item["weight_percent"] for item in mix if item["access_mode"] == "read_write")
    application = parse_qs(urlparse(root.findtext("url", "")).query).get("ApplicationName")
    name = application[0] if application else path.stem.replace("sample_", "").replace("_config", "")
    return {
        "name": name, "driver": "BenchBase", "config_path": str(path.resolve()),
        "database": _jdbc_database(root.findtext("url", "")),
        "scale_factor": float(root.findtext("scalefactor", "0")), "terminals": int(root.findtext("terminals", "0")),
        "warmup_seconds": int(root.findtext(".//works/work/warmup", "0")),
        "measurement_seconds": int(root.findtext(".//works/work/time", "0")),
        "rate": root.findtext(".//works/work/rate"), "isolation": root.findtext("isolation"),
        "batch_size": int(root.findtext("batchsize", "0")), "transaction_mix": mix,
        "read_write_transaction_percent": read_write, "read_only_transaction_percent": 100.0 - read_write,
    }


def _postgres_profile(host: str, port: int, user: str, password: str, database: str) -> dict[str, Any]:
    with psycopg2.connect(host=host, port=port, database="postgres", user=user, password=password) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT version(), current_setting('server_version_num'), current_setting('data_directory')")
            version, version_num, data_directory = cursor.fetchone()
            cursor.execute("SELECT name, setting, unit FROM pg_settings WHERE name = ANY(%s) ORDER BY name", (list(_KEY_SETTINGS),))
            settings = {name: {"setting": value, "unit": unit} for name, value, unit in cursor.fetchall()}
            cursor.execute("SELECT datname, pg_database_size(datname) FROM pg_database WHERE datname LIKE 'benchbase_%' ORDER BY datname")
            sizes = {name: size for name, size in cursor.fetchall()}
            cursor.execute("SELECT datname,xact_commit,xact_rollback,tup_returned,tup_fetched,tup_inserted,tup_updated,tup_deleted,blks_read,blks_hit FROM pg_stat_database WHERE datname=%s", (database,))
            columns = [item.name for item in cursor.description]
            cumulative = dict(zip(columns, cursor.fetchone()))
    tables = []
    with psycopg2.connect(host=host, port=port, database=database, user=user, password=password) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT relname,n_live_tup,pg_total_relation_size(relid) FROM pg_stat_user_tables ORDER BY pg_total_relation_size(relid) DESC")
            tables = [{"table": name, "estimated_rows": rows, "total_bytes": size} for name, rows, size in cursor.fetchall()]
    return {
        "engine": "PostgreSQL", "version": version, "version_num": version_num,
        "host": host, "port": port, "user": user, "data_directory": data_directory,
        "settings": settings, "database_sizes_bytes": sizes, "target_database": database,
        "target_tables": tables, "cumulative_target_database_counters": cumulative,
    }


_KEY_SETTINGS = {
    "shared_buffers", "work_mem", "maintenance_work_mem", "effective_cache_size", "max_connections",
    "max_worker_processes", "max_parallel_workers", "max_parallel_workers_per_gather", "wal_buffers",
    "checkpoint_timeout", "max_wal_size", "effective_io_concurrency", "huge_pages", "synchronous_commit",
}


def _typed_default(vartype: str, value: str) -> Any:
    if vartype == "integer":
        return int(value)
    if vartype == "real":
        return float(value)
    if vartype == "bool":
        return value.lower() in {"on", "true", "yes", "1"}
    return value


def _category(category: str, name: str) -> str:
    text = f"{category} {name}".lower()
    if any(word in text for word in ("memory", "buffer", "cache")):
        return "memory"
    if any(word in text for word in ("wal", "checkpoint", "writer", "vacuum", "disk", "file")):
        return "storage"
    if any(word in text for word in ("planner", "cost", "statistics", "collapse", "jit")):
        return "optimizer"
    if any(word in text for word in ("connection", "lock", "worker", "standby", "replication")):
        return "concurrency"
    return "execution"


def _jdbc_database(url: str) -> str:
    match = re.search(r"/([^/?]+)(?:\?|$)", url)
    return match.group(1) if match else ""


def _run(command: list[str]) -> str:
    return subprocess.run(command, text=True, capture_output=True, check=True).stdout


def _parse_key_values(text: str) -> dict[str, str]:
    return {key.strip(): value.strip() for line in text.splitlines() if ":" in line for key, value in [line.split(":", 1)]}


def _parse_meminfo(text: str) -> dict[str, int]:
    return {key.rstrip(":"): int(value) for line in text.splitlines() for key, value, *_ in [line.split()]}


def _read_optional(path: str) -> str | None:
    candidate = Path(path)
    return candidate.read_text().strip() if candidate.exists() else None
