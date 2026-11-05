#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

install -m 755 "$ROOT/scripts/codex_proxy.py" /usr/local/bin/codex-proxy
if [[ ! -f /etc/codex-proxy.env ]]; then
  install -m 644 "$ROOT/configs/codex-proxy.env.example" /etc/codex-proxy.env
  echo "Edit /etc/codex-proxy.env with the host IP and ports."
fi

cat > /etc/profile.d/99-codex-proxy.sh <<'PROFILE'
if [ -x /usr/local/bin/codex-proxy ]; then
  eval "$(/usr/local/bin/codex-proxy env 2>/dev/null)"
fi
PROFILE
chmod 755 /etc/profile.d/99-codex-proxy.sh

if ! grep -q '99-codex-proxy' /root/.bashrc 2>/dev/null; then
  printf '\n# Codex host proxy autostart\n[ -r /etc/profile.d/99-codex-proxy.sh ] && . /etc/profile.d/99-codex-proxy.sh\n' >> /root/.bashrc
fi

cat > /root/.vscode-server/server-env-setup <<'SERVER'
#!/usr/bin/env bash
timeout 6 /usr/local/bin/codex-proxy tunnel-start >/dev/null 2>&1 || true
if [ -x /usr/local/bin/codex-proxy ]; then
  eval "$(/usr/local/bin/codex-proxy env --refresh 2>/dev/null)"
fi
SERVER
chmod 755 /root/.vscode-server/server-env-setup

if [[ ! -f /root/.ssh/codex_proxy_ed25519 ]]; then
  mkdir -p /root/.ssh
  chmod 700 /root/.ssh
  ssh-keygen -t ed25519 -N '' -f /root/.ssh/codex_proxy_ed25519 -C codex-proxy-container >/dev/null
fi
chmod 600 /root/.ssh/codex_proxy_ed25519
chmod 644 /root/.ssh/codex_proxy_ed25519.pub
echo "Installed. Add this public key to the host's ~/.ssh/authorized_keys:"
cat /root/.ssh/codex_proxy_ed25519.pub
