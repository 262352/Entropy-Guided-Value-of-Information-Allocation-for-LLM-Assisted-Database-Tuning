# Codex / container host-proxy setup

## Diagnosis

From the container (`172.17.0.11`, gateway `172.17.0.1`):

| target | result |
|---|---|
| `218.194.39.27:7980` | **Connection refused** |
| `172.17.0.1:7980` | timeout |
| `218.194.39.27:7890` | **open**, proxy works (`chatgpt.com` -> 200) |
| `172.17.0.1:22` | open (SSH), key auth required |
| Codex process env | **no HTTP(S)_PROXY**; shell proxy never reached the extension |

So the immediate Codex failure is not DNS: the Codex app-server inherited no
proxy variables, and the requested host port 7980 is not reachable from the
container. Port 7890 is reachable and works. 7980 is probably bound to the
host's `127.0.0.1` only, or not started.

## Installed autostart

- `/usr/local/bin/codex-proxy` — detect/apply proxy, optional SSH tunnel.
- `/etc/codex-proxy.env` — host IP/ports/SSH settings.
- `/etc/profile.d/99-codex-proxy.sh` — login/interactive shell autostart.
- `~/.bashrc` — non-login interactive shell autostart.
- `~/.vscode-server/server-env-setup` — exports proxy into the VS Code server
  and extension host before startup; also tries the SSH tunnel.
- `/root/.ssh/codex_proxy_ed25519(.pub)` — key for the optional tunnel.

The proxy is chosen in this order and cached for 5 minutes:

1. `218.194.39.27:7980`
2. `172.17.0.1:7980`
3. `127.0.0.1:7980` (SSH tunnel)
4. `218.194.39.27:7890`
5. `172.17.0.1:7890`
6. direct

Current result: **falls back to `218.194.39.27:7890`**, because 7980 is closed.

## Make 7980 work (host-side, choose one)

### A. Expose the host proxy on 7980

On the host, configure the proxy to listen on `0.0.0.0:7980` (Clash:
`allow-lan: true`, `bind-address: '*'`, `port: 7980`) and allow the Docker
subnet `172.17.0.0/16` through the firewall. Verify from the container:

```bash
curl -x http://218.194.39.27:7980 -I https://chatgpt.com
```

Then on the container:

```bash
/usr/local/bin/codex-proxy detect --refresh   # should pick 7980
```

### B. SSH tunnel to host-localhost:7980 (recommended if 7980 is local-only)

On the host, add this public key to the SSH user's
`~/.ssh/authorized_keys` (the key generated in this container):

```text
<public key printed by scripts/install_codex_proxy.sh>
```

If the host SSH user is not `root`, edit `SSH_USER=` in
`/etc/codex-proxy.env`. Then in the container:

```bash
/usr/local/bin/codex-proxy tunnel-start
/usr/local/bin/codex-proxy detect --refresh
```

The tunnel forwards container `127.0.0.1:7980` to host `127.0.0.1:7980`.
`server-env-setup` will attempt it automatically on the next VS Code server
start once the key is authorized.

### C. Keep using 7890 now

`218.194.39.27:7890` is already reachable and returns HTTP 200 through the
proxy. The autostart uses it as fallback, so Codex can work immediately after
its environment is refreshed.

## Apply to the running Codex extension

The already-running Codex process (`~/.vscode-server/extensions/openai.chatgpt...`)
does not inherit changes made after it started. Restart the VS Code server /
reopen the remote window so `server-env-setup` runs. Check:

```bash
tr '\0' '\n' < /proc/$(pgrep -f 'bin/linux-x86_64/codex' | head -1)/environ | grep -i proxy
```

For a terminal, open a new shell and check:

```bash
codex-proxy status
curl -x "$HTTPS_PROXY" -I https://chatgpt.com
```

## Important

- The private SSH key is in this container only; do not commit it.
- The public key is safe to paste on the host.
- If 7980 is not running at all on the host, neither direct nor tunnel can help;
  start it or use 7890.
- The container currently has **no systemd** (PID 1 is bash), so autostart is
  through `server-env-setup`, `/etc/profile.d`, and `~/.bashrc`, not systemd.
