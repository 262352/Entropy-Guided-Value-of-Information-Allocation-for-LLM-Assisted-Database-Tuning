# Codex container proxy (host port 7890)

The container reaches the host proxy at:

```text
http://<HOST_IP>:7890
```

This is the supported configuration. Port 7980 is not used.

## Install

```bash
./scripts/install_codex_proxy.sh
# edit /etc/codex-proxy.env: HOST_IP=<your host reachable IPv4>
codex-proxy status
```

## Autostart

The installer writes:

- `/usr/local/bin/codex-proxy` — detect/apply proxy.
- `/etc/codex-proxy.env` — host IP and `PROXY_PORT=7890`.
- `/etc/profile.d/99-codex-proxy.sh` — login shells.
- `~/.bashrc` — interactive shells.
- `~/.vscode-server/server-env-setup` — VS Code server/Codex extension.

Codex extension processes started before the env injection need a VS Code
server restart (`Remote: Kill VS Code Server on Host` or reopen the window).

## Verify

```bash
codex-proxy status
bash -lc 'echo $HTTPS_PROXY'
curl -x "$HTTPS_PROXY" -I https://chatgpt.com
```

A proxy is selected only after an actual HTTPS CONNECT test succeeds; if the
host proxy is down, the script falls back to direct and reports it.
