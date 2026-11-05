#!/usr/bin/env python3
"""Detect and apply the host proxy for Codex / VS Code / shells.

Priority:
  1. host-ip:7980
  2. docker-gateway:7980
  3. 127.0.0.1:7980 (SSH tunnel)
  4. host-ip:7890
  5. docker-gateway:7890
  6. direct

Usage:
  codex-proxy status
  codex-proxy detect [--refresh]
  codex-proxy env [--refresh]
  codex-proxy tunnel-start
  codex-proxy tunnel-status
  codex-proxy tunnel-stop
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

CONFIG = Path("/etc/codex-proxy.env")
CACHE = Path("/run/codex-proxy.cache.json")
CACHE_TTL = 300


def load_config() -> dict[str, str]:
    data = {
        "HOST_IP": "218.194.39.27",
        "HOST_GATEWAY": "172.17.0.1",
        "PROXY_PORT": "7980",
        "FALLBACK_PORT": "7890",
        "SSH_HOST": "172.17.0.1",
        "SSH_PORT": "22",
        "SSH_USER": "root",
        "SSH_KEY": "/root/.ssh/codex_proxy_ed25519",
    }
    if CONFIG.is_file():
        for line in CONFIG.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    for key in list(data):
        data[key] = os.environ.get(key, data[key])
    return data


def candidates(cfg: dict[str, str]) -> list[tuple[str, str]]:
    host = cfg["HOST_IP"]
    gateway = cfg["HOST_GATEWAY"]
    preferred = cfg["PROXY_PORT"]
    fallback = cfg["FALLBACK_PORT"]
    return [
        (f"http://{host}:{preferred}", f"host-ip:{preferred}"),
        (f"http://{gateway}:{preferred}", f"docker-gateway:{preferred}"),
        (f"http://127.0.0.1:{preferred}", f"ssh-tunnel:{preferred}"),
        (f"http://{host}:{fallback}", f"host-ip:{fallback}"),
        (f"http://{gateway}:{fallback}", f"docker-gateway:{fallback}"),
    ]


def probe(url: str, timeout: int = 8) -> tuple[bool, str]:
    """Return (works, detail) using curl with an HTTPS CONNECT target."""
    try:
        completed = subprocess.run(
            ["curl", "-x", url, "--max-time", str(timeout), "-sS",
             "-o", "/dev/null", "-w", "%{http_code}", "https://chatgpt.com"],
            capture_output=True, text=True, timeout=timeout + 2,
        )
    except Exception as error:
        return False, f"curl-error:{type(error).__name__}"
    code = (completed.stdout or "").strip()
    if completed.returncode == 0 and code not in {"", "000"}:
        return True, f"http={code}"
    detail = (completed.stderr or "").strip().replace("\n", " ")[:120]
    return False, f"rc={completed.returncode},http={code},err={detail}"


def detect(refresh: bool = False) -> dict:
    if not refresh and CACHE.is_file():
        try:
            cached = json.loads(CACHE.read_text(encoding="utf-8"))
            if time.time() - float(cached.get("ts", 0)) < CACHE_TTL:
                return cached
        except Exception:
            pass
    cfg = load_config()
    checks = []
    chosen = None
    for url, source in candidates(cfg):
        ok, detail = probe(url)
        checks.append({"url": url, "source": source, "ok": ok, "detail": detail})
        if ok:
            chosen = {"url": url, "source": source, "detail": detail}
            break
    result = {
        "ts": time.time(),
        "proxy": chosen["url"] if chosen else "",
        "source": chosen["source"] if chosen else "direct",
        "detail": chosen["detail"] if chosen else "no reachable proxy",
        "checks": checks,
    }
    try:
        CACHE.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass
    return result


def env_lines(result: dict) -> str:
    proxy = result.get("proxy", "")
    no_proxy = "localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,.local"
    lines = []
    if proxy:
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            lines.append(f"export {name}='{proxy}'")
    else:
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            lines.append(f"unset {name}")
    lines.append(f"export NO_PROXY='{no_proxy}'")
    lines.append(f"export no_proxy='{no_proxy}'")
    lines.append("export NODE_USE_ENV_PROXY=1")
    lines.append(f"export CODEX_PROXY_SOURCE='{result.get('source', 'direct')}'")
    return "\n".join(lines)


def tunnel_start() -> int:
    cfg = load_config()
    key = Path(cfg["SSH_KEY"])
    if not key.is_file():
        print(f"missing SSH key: {key}; run ssh-keygen first", file=sys.stderr)
        return 2
    # If something already listens on 7980 locally, do not start a second tunnel.
    if probe(f"http://127.0.0.1:{cfg['PROXY_PORT']}")[0]:
        print(f"127.0.0.1:{cfg['PROXY_PORT']} already works")
        return 0
    command = [
        "ssh", "-fN",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-o", "StrictHostKeyChecking=accept-new",
        "-i", str(key),
        "-p", cfg["SSH_PORT"],
        "-L", f"127.0.0.1:{cfg['PROXY_PORT']}:127.0.0.1:{cfg['PROXY_PORT']}",
        f"{cfg['SSH_USER']}@{cfg['SSH_HOST']}",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=20)
    if completed.returncode != 0:
        print(completed.stderr.strip() or completed.stdout.strip(), file=sys.stderr)
        print("Add the public key to the host's ~/.ssh/authorized_keys, then retry.", file=sys.stderr)
        return completed.returncode
    time.sleep(1)
    ok, detail = probe(f"http://127.0.0.1:{cfg['PROXY_PORT']}")
    print(f"tunnel {'ok' if ok else 'failed'}: {detail}")
    return 0 if ok else 1


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    refresh = "--refresh" in sys.argv[2:]
    result = detect(refresh=refresh)
    if command == "env":
        print(env_lines(result))
        return 0
    if command == "detect":
        print(json.dumps(result, indent=2))
        return 0 if result.get("proxy") else 1
    if command == "status":
        print(json.dumps(result, indent=2))
        print("\n# shell exports")
        print(env_lines(result))
        return 0 if result.get("proxy") else 1
    if command == "tunnel-start":
        return tunnel_start()
    if command == "tunnel-status":
        cfg = load_config()
        ok, detail = probe(f"http://127.0.0.1:{cfg['PROXY_PORT']}")
        print(json.dumps({"tunnel": ok, "detail": detail}, indent=2))
        return 0 if ok else 1
    if command == "tunnel-stop":
        cfg = load_config()
        subprocess.run(["pkill", "-f", f"-L 127.0.0.1:{cfg['PROXY_PORT']}:127.0.0.1:{cfg['PROXY_PORT']}"], check=False)
        print("tunnel stopped")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
