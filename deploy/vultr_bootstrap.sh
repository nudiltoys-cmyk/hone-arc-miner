#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/nudiltoys-cmyk/hone-arc-miner"
REPO_BRANCH="main"
SOLVER_COMMIT="e4c0d72a445da95d6bc1e4edf4b6d298763fcbf2"
HOTKEY="5Cvvp52bNd4MLiTzhkEDt5NTGL89xWvMvP8maswDCwVuvqBb"
PUBLIC_KEY="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPDQP4NaSwmHjQ5td01U4QsPhlBVRi3x8R07lVX5DStX hone-sn5-vps"

export DEBIAN_FRONTEND=noninteractive

mkdir -p /root/.ssh
chmod 700 /root/.ssh
printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys

apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  git \
  openssh-server \
  python3 \
  python3-pip \
  python3-venv

systemctl enable --now ssh

mkdir -p /opt
if [ -d /opt/hone-arc-miner/.git ]; then
  git -C /opt/hone-arc-miner fetch --all --prune
  git -C /opt/hone-arc-miner checkout "$REPO_BRANCH"
  git -C /opt/hone-arc-miner pull --ff-only origin "$REPO_BRANCH"
else
  rm -rf /opt/hone-arc-miner
  git clone --branch "$REPO_BRANCH" "$REPO_URL" /opt/hone-arc-miner
fi

python3 -m venv /opt/hone-miner-venv
/opt/hone-miner-venv/bin/pip install --upgrade pip
/opt/hone-miner-venv/bin/pip install -r /opt/hone-arc-miner/miner-server/requirements.txt

cat >/etc/systemd/system/hone-miner.service <<SERVICE
[Unit]
Description=Hone SN5 miner info server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/hone-arc-miner/miner-server
Environment=MINER_REPO_URL=$REPO_URL
Environment=MINER_REPO_BRANCH=$REPO_BRANCH
Environment=MINER_REPO_COMMIT=$SOLVER_COMMIT
Environment=MINER_REPO_PATH=solver
Environment=MINER_WEIGHT_CLASS=1xH200
Environment=MINER_USE_VLLM=false
Environment=MINER_HOTKEY_SS58=$HOTKEY
Environment=MINER_VERSION=e4c0d72
ExecStart=/opt/hone-miner-venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8091
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
systemctl enable hone-miner.service
systemctl restart hone-miner.service

if command -v ufw >/dev/null 2>&1; then
  ufw allow OpenSSH || true
  ufw allow 8091/tcp || true
fi

for _ in 1 2 3 4 5 6 7 8 9 10; do
  if curl -fsS http://127.0.0.1:8091/health; then
    echo
    echo "Hone miner bootstrap complete."
    exit 0
  fi
  sleep 1
done

curl -fsS http://127.0.0.1:8091/health
echo
echo "Hone miner bootstrap complete."
