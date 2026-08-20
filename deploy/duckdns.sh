#!/usr/bin/env bash
# Keep the DuckDNS A record pointed at this instance.
#
# Oracle public IPs are ephemeral unless reserved, and a changed IP breaks both
# DNS and the TLS certificate. Reserving the IP is the real fix; this is the
# belt to that braces.
set -euo pipefail

: "${DUCKDNS_DOMAIN:?set DUCKDNS_DOMAIN (the subdomain only, no .duckdns.org)}"
: "${DUCKDNS_TOKEN:?set DUCKDNS_TOKEN}"

# Empty ip= makes DuckDNS use the source address of this request.
response=$(curl -fsS \
	"https://www.duckdns.org/update?domains=${DUCKDNS_DOMAIN}&token=${DUCKDNS_TOKEN}&ip=")

if [ "$response" != "OK" ]; then
	echo "duckdns update failed: ${response}" >&2
	exit 1
fi
echo "duckdns update OK"
