#!/usr/bin/env bash
# Idempotent provisioning for the AutoAce host.
#
# Supports two distros, detected at runtime:
#   - Ubuntu 24.04 aarch64 (apt), the original target.
#   - Oracle Linux 9 aarch64 (dnf) - the distro actually used for the live
#     deployment (verified on Oracle Linux Server 9.8, aarch64, 2 OCPU,
#     ~10 GiB usable RAM).
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
# speechbrain resolves `savedir` against the CWD rather than HF_HOME, so ECAPA
# needs its own explicit absolute cache dir or every process re-downloads it.
MODELS_DIR=/opt/autoace/models
ENV_FILE="/etc/autoace.env"

# App user is never hardcoded to "ubuntu": default to the user who invoked
# sudo, fall back to the current user, and let the operator override.
APP_USER="${AUTOACE_APP_USER:-${SUDO_USER:-$(id -un)}}"

log() {
	printf '==> %s\n' "$*"
}

# --------------------------------------------------------------- distro detect
if command -v dnf >/dev/null 2>&1; then
	PKG=dnf
elif command -v apt-get >/dev/null 2>&1; then
	PKG=apt
else
	echo "Unsupported distro: neither dnf nor apt-get found" >&2
	exit 1
fi
log "Detected package manager: $PKG (app user: $APP_USER)"

# ---------------------------------------------------------------- packages
# ffmpeg and libsndfile are intentionally NOT installed. soundfile bundles
# libsndfile in its wheel (_soundfile_data/), and PyAV (used by
# faster-whisper) bundles the ffmpeg libraries (av.libs, ~63 MB). Nothing in
# this codebase shells out to an ffmpeg binary, so the package is dead
# weight - and on Oracle Linux 9 it isn't even in the base repos (needs
# EPEL + RPM Fusion), so dropping it removes the only hard package blocker.
log "Installing base packages"
if [ "$PKG" = dnf ]; then
	sudo dnf install -y \
		python3.12 \
		python3.12-pip \
		git \
		curl
else
	sudo apt-get update -y
	sudo apt-get install -y \
		python3.12-venv \
		python3-pip \
		git \
		curl \
		debian-keyring \
		debian-archive-keyring \
		apt-transport-https \
		gnupg
fi

# Needed for the idempotent iptables firewall rules below
# (netfilter-persistent save). Only relevant on the apt/iptables path -
# firewalld (dnf path) manages its own persistence.
if [ "$PKG" = apt ] && ! command -v netfilter-persistent >/dev/null 2>&1; then
	sudo apt-get install -y iptables-persistent
fi

# -------------------------------------------------------------------- caddy
if ! command -v caddy >/dev/null 2>&1; then
	if [ "$PKG" = apt ]; then
		log "Installing Caddy from its official apt repo"
		curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
			| sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
		curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
			| sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
		sudo apt-get update -y
		sudo apt-get install -y caddy
	else
		# Caddy is not packaged for Oracle Linux 9. Download the static
		# binary for the current arch from the GitHub latest release.
		log "Installing Caddy static binary from the latest GitHub release"
		ARCH=$(uname -m)
		case "$ARCH" in
			aarch64) CADDY_ARCH=arm64 ;;
			x86_64) CADDY_ARCH=amd64 ;;
			*)
				echo "unsupported arch $ARCH" >&2
				exit 1
				;;
		esac
		CADDY_URL=$(curl -fsSL https://api.github.com/repos/caddyserver/caddy/releases/latest \
			| grep -o "https://[^\"]*linux_${CADDY_ARCH}\.tar\.gz" | head -1)
		curl -fsSL "$CADDY_URL" -o /tmp/caddy.tgz
		tar -xzf /tmp/caddy.tgz -C /tmp caddy
		sudo install -m 0755 /tmp/caddy /usr/bin/caddy
		rm -f /tmp/caddy.tgz /tmp/caddy

		# No packaged unit on this distro, so we own the systemd unit and
		# the caddy user/dirs ourselves. Idempotent.
		if ! id caddy >/dev/null 2>&1; then
			log "Creating caddy system user"
			sudo useradd --system --home /var/lib/caddy --shell /usr/sbin/nologin caddy
		fi
		sudo mkdir -p /var/lib/caddy /etc/caddy
		sudo chown caddy:caddy /var/lib/caddy

		log "Writing caddy.service (upstream shape; not packaged on dnf distros)"
		sudo tee /etc/systemd/system/caddy.service >/dev/null <<'CADDYUNITEOF'
[Unit]
Description=Caddy
Documentation=https://caddyserver.com/docs/
After=network.target network-online.target
Requires=network-online.target

[Service]
Type=notify
User=caddy
Group=caddy
ExecStart=/usr/bin/caddy run --environ --config /etc/caddy/Caddyfile
ExecReload=/usr/bin/caddy reload --config /etc/caddy/Caddyfile --force
TimeoutStopSec=5s
LimitNOFILE=1048576
PrivateTmp=true
ProtectSystem=full
AmbientCapabilities=CAP_NET_BIND_SERVICE

[Install]
WantedBy=multi-user.target
CADDYUNITEOF
	fi
else
	log "Caddy already installed, skipping"
fi

# ----------------------------------------------------------------- firewall
# Oracle's stock images ship host firewall rules that DROP everything except
# SSH. The VCN security list can look perfectly correct while the host's own
# firewall still refuses the connection - this is the single most common way
# this deploy fails. Port 80 must stay open PERMANENTLY, not just for first
# issuance: Caddy's ACME renewals reuse it, not only HTTPS 443.
log "Opening 80/443 in the host firewall"
if command -v firewall-cmd >/dev/null 2>&1 && sudo systemctl is-active --quiet firewalld; then
	# firewalld (Oracle Linux 9 default): manages nftables under the hood,
	# so poking iptables directly here would be a no-op. --permanent +
	# --reload is already idempotent.
	sudo firewall-cmd --permanent --add-service=http --add-service=https
	sudo firewall-cmd --reload
else
	# iptables path (Ubuntu default).
	for port in 80 443; do
		if ! sudo iptables -C INPUT -p tcp --dport "$port" -j ACCEPT 2>/dev/null; then
			sudo iptables -I INPUT 6 -p tcp --dport "$port" -j ACCEPT
		fi
	done
	if command -v netfilter-persistent >/dev/null 2>&1; then
		sudo netfilter-persistent save
	else
		log "netfilter-persistent not found, skipping rule persistence"
	fi
fi

# --------------------------------------------------------------------- swap
# Insurance for the measured 5.67 GiB transient worker peak against a 3.91
# GiB steady state. Swapping is slow, but slow beats an OOM kill of the
# worker mid-job. Oracle Linux 9 ships 4 GiB already active at /.swapfile
# (note the leading dot, unlike Ubuntu's /swapfile), so check for existing
# active swap rather than a hardcoded path, or this would create a
# redundant second swapfile.
EXISTING_SWAP_BYTES=$(swapon --show=SIZE --noheadings --bytes 2>/dev/null | awk '{sum += $1} END {print sum+0}')
if [ "$EXISTING_SWAP_BYTES" -lt 2000000000 ]; then
	log "Creating 4 GiB swapfile"
	sudo fallocate -l 4G /swapfile
	sudo chmod 600 /swapfile
	sudo mkswap /swapfile
	sudo swapon /swapfile
	echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
else
	log "At least 2 GiB of swap already active, skipping"
fi

# ---------------------------------------------------------- data directories
log "Creating /opt/autoace directories"
sudo mkdir -p /opt/autoace
sudo chown "$APP_USER:$APP_USER" /opt/autoace
sudo -u "$APP_USER" mkdir -p "$DATA_DIR" "$HF_DIR" "$MODELS_DIR"

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
sudo -u "$APP_USER" "$VENV_DIR/bin/pip" install --quiet --upgrade pip
# CPU-only torch FIRST. On aarch64 Linux the default PyPI torch is the CUDA
# build and drags in ~5 GB of nvidia_* wheels that are dead weight here.
# Windows defaults to CPU, which is why this never showed up in development.
sudo -u "$APP_USER" "$VENV_DIR/bin/pip" install --index-url https://download.pytorch.org/whl/cpu \
	'torch>=2.13' 'torchaudio>=2.11'
# Everything else from PyPI. torch/torchaudio are already satisfied, so the
# resolver will not pull the CUDA build.
sudo -u "$APP_USER" "$VENV_DIR/bin/pip" install -r "$REPO_DIR/requirements.txt"

# ---------------------------------------------------------------- unit files
log "Installing systemd units"
# The in-repo units default to User=ubuntu; rewrite that line to the actual
# app user derived above rather than hardcoding it (or requiring a template).
sed "s/^User=.*/User=${APP_USER}/" "$REPO_DIR/deploy/autoace-web.service" \
	| sudo tee /etc/systemd/system/autoace-web.service >/dev/null
sed "s/^User=.*/User=${APP_USER}/" "$REPO_DIR/deploy/autoace-worker.service" \
	| sudo tee /etc/systemd/system/autoace-worker.service >/dev/null
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
# Only the hostname, NOT EnvironmentFile=/etc/autoace.env. Caddy needs one
# variable; handing it the whole file would put OPENAI_API_KEY,
# AUTOACE_PASSWORD and DUCKDNS_TOKEN into the environment of a process that
# has no use for any of them. The hostname itself is public.
sudo tee /etc/systemd/system/caddy.service.d/override.conf >/dev/null <<DROPINEOF
[Service]
Environment=AUTOACE_HOSTNAME=${AUTOACE_HOSTNAME}
DROPINEOF

# ---------------------------------------------------------------- model warm-up
# Downloads ~5 GiB of weights once, ever. HF_HOME lives on the boot volume so
# this survives restarts and future code deploys without re-downloading.
log "Warming model cache (~5 GiB total on first run; re-runs are fast no-ops)"
(
	cd "$REPO_DIR"
	sudo -u "$APP_USER" env HF_HOME="$HF_DIR" AUTOACE_MODELS_DIR="$MODELS_DIR" "$VENV_DIR/bin/python" - <<'PY'
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
