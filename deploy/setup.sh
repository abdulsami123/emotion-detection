#!/usr/bin/env bash
# Idempotent provisioning for the AutoAce Oracle ARM VM
# (VM.Standard.A1.Flex, 2 OCPU / 12 GiB, Ubuntu 24.04 aarch64, Python 3.12).
#
# Safe to re-run: every step checks existing state before acting. Run this
# script as a user with passwordless (or interactive) sudo, e.g.:
#   /opt/autoace/repo/deploy/setup.sh
set -euo pipefail

REPO_URL="https://github.com/abdulsami123/emotion-detection.git"
REPO_DIR="/opt/autoace/repo"
VENV_DIR="/opt/autoace/venv"
DATA_DIR="/opt/autoace/data"
HF_DIR="/opt/autoace/hf"
ENV_FILE="/etc/autoace.env"
APP_USER="ubuntu"

log() {
	printf '==> %s\n' "$*"
}

# ---------------------------------------------------------------- packages
log "Installing base packages"
sudo apt-get update -y
sudo apt-get install -y \
	python3.12-venv \
	python3-pip \
	git \
	ffmpeg \
	libsndfile1 \
	curl \
	debian-keyring \
	debian-archive-keyring \
	apt-transport-https \
	gnupg

# Needed for the idempotent firewall rules below (netfilter-persistent save).
# Oracle's stock Ubuntu images normally ship this already; installed here
# defensively in case a bare image does not.
if ! command -v netfilter-persistent >/dev/null 2>&1; then
	sudo apt-get install -y iptables-persistent
fi

# -------------------------------------------------------------------- caddy
if ! command -v caddy >/dev/null 2>&1; then
	log "Installing Caddy from its official apt repo"
	curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
		| sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
	curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
		| sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
	sudo apt-get update -y
	sudo apt-get install -y caddy
else
	log "Caddy already installed, skipping"
fi

# ----------------------------------------------------------------- firewall
# Oracle's stock Ubuntu images ship iptables rules that DROP everything
# except SSH. The VCN security list can look perfectly correct while the
# host's own netfilter rules still refuse the connection - this is the single
# most common way this deploy fails. Port 80 must stay open PERMANENTLY, not
# just for first issuance: Caddy's ACME renewals reuse it, not only HTTPS 443.
log "Opening 80/443 in the host firewall"
for port in 80 443; do
	if ! sudo iptables -C INPUT -p tcp --dport "$port" -j ACCEPT 2>/dev/null; then
		sudo iptables -I INPUT 6 -p tcp --dport "$port" -j ACCEPT
	fi
done
sudo netfilter-persistent save

# --------------------------------------------------------------------- swap
# Insurance for the measured 5.67 GiB transient worker peak against a 3.91
# GiB steady state on a 12 GiB box. Swapping is slow, but slow beats an OOM
# kill of the worker mid-job.
if [ ! -f /swapfile ]; then
	log "Creating 4 GiB swapfile"
	sudo fallocate -l 4G /swapfile
	sudo chmod 600 /swapfile
	sudo mkswap /swapfile
	sudo swapon /swapfile
	echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
else
	log "Swapfile already present, skipping"
fi

# ---------------------------------------------------------- data directories
log "Creating /opt/autoace directories"
sudo mkdir -p /opt/autoace
sudo chown "$APP_USER:$APP_USER" /opt/autoace
sudo -u "$APP_USER" mkdir -p "$DATA_DIR" "$HF_DIR"

# ----------------------------------------------------------------------- repo
if [ -d "$REPO_DIR/.git" ]; then
	log "Updating existing repo checkout"
	sudo -u "$APP_USER" git -C "$REPO_DIR" pull --ff-only
else
	log "Cloning repo"
	sudo -u "$APP_USER" git clone "$REPO_URL" "$REPO_DIR"
fi

# ----------------------------------------------------------------------- venv
if [ ! -x "$VENV_DIR/bin/python" ]; then
	log "Creating virtualenv"
	sudo -u "$APP_USER" python3.12 -m venv "$VENV_DIR"
fi
log "Installing Python dependencies (this takes a while on first run)"
sudo -u "$APP_USER" "$VENV_DIR/bin/pip" install --upgrade pip
sudo -u "$APP_USER" "$VENV_DIR/bin/pip" install -r "$REPO_DIR/requirements.txt"

# ---------------------------------------------------------------- unit files
log "Installing systemd units"
sudo cp "$REPO_DIR/deploy/autoace-web.service" /etc/systemd/system/autoace-web.service
sudo cp "$REPO_DIR/deploy/autoace-worker.service" /etc/systemd/system/autoace-worker.service
sudo cp "$REPO_DIR/deploy/duckdns.service" /etc/systemd/system/duckdns.service
sudo cp "$REPO_DIR/deploy/duckdns.timer" /etc/systemd/system/duckdns.timer
sudo chmod +x "$REPO_DIR/deploy/duckdns.sh"

# ------------------------------------------------------------------- secrets
# /etc/autoace.env is created by the operator, never by this script and never
# committed - it holds the OpenAI key and dashboard password.
if [ ! -f "$ENV_FILE" ]; then
	cat <<MSGEOF
$ENV_FILE is missing. Create it now (mode 0600, operator-owned secrets -
none of these values are written by this script), then re-run setup.sh:

  sudo install -m 600 /dev/null $ENV_FILE
  sudo tee $ENV_FILE >/dev/null <<'ENVEOF'
OPENAI_API_KEY=sk-replace-me
AUTOACE_USER=admin
AUTOACE_PASSWORD=replace-me
AUTOACE_HOSTNAME=yoursubdomain.duckdns.org
DUCKDNS_DOMAIN=yoursubdomain
DUCKDNS_TOKEN=replace-me
ENVEOF

MSGEOF
	exit 1
fi

set -a
# shellcheck source=/dev/null
source "$ENV_FILE"
set +a
: "${AUTOACE_HOSTNAME:?AUTOACE_HOSTNAME must be set in $ENV_FILE}"

# --------------------------------------------------------------------- caddy
log "Installing Caddyfile"
sudo mkdir -p /etc/caddy
sudo cp "$REPO_DIR/deploy/Caddyfile" /etc/caddy/Caddyfile

log "Passing AUTOACE_HOSTNAME to Caddy via a systemd drop-in"
sudo mkdir -p /etc/systemd/system/caddy.service.d
sudo tee /etc/systemd/system/caddy.service.d/override.conf >/dev/null <<DROPINEOF
[Service]
EnvironmentFile=$ENV_FILE
DROPINEOF

# ---------------------------------------------------------------- model warm-up
# Downloads ~5 GiB of weights once, ever. HF_HOME lives on the boot volume so
# this survives restarts and future code deploys without re-downloading.
log "Warming model cache (~5 GiB total on first run; re-runs are fast no-ops)"
(
	cd "$REPO_DIR"
	sudo -u "$APP_USER" env HF_HOME="$HF_DIR" "$VENV_DIR/bin/python" - <<'PY'
print("loading whisper (asr)...", flush=True)
from autoace.asr import _load_model as _load_asr_model
_load_asr_model()

print("loading ecapa encoder (diarize)...", flush=True)
from autoace.diarize import _load_encoder
_load_encoder()

print("loading SER model...", flush=True)
from autoace.ser import _load_model as _load_ser_model
_load_ser_model()

print("loading AST tagging model...", flush=True)
from autoace.tagging import _load_ast
_load_ast()

print("loading NLI zero-shot classifier (bart-large-mnli fallback)...", flush=True)
from autoace.tone_nli import _load_classifier
_load_classifier()

print("loading SQUIM quality model...", flush=True)
from autoace.quality import _load_squim
_load_squim()

print("model warm-up complete", flush=True)
PY
)

# --------------------------------------------------------------- enable & start
log "Reloading systemd"
sudo systemctl daemon-reload

log "Running one DuckDNS update now, so DNS is correct before Caddy requests a certificate"
sudo systemctl start duckdns.service

log "Enabling and starting services"
sudo systemctl enable --now duckdns.timer
sudo systemctl enable --now autoace-worker.service
sudo systemctl enable --now autoace-web.service
sudo systemctl enable --now caddy.service

log "Setup complete."
echo "AutoAce should be reachable at: https://${AUTOACE_HOSTNAME}"
echo "Tail worker logs with: journalctl -u autoace-worker -f"
