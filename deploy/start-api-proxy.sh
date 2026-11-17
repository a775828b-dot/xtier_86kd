#!/usr/bin/env bash
set -euo pipefail

until [ -x /data/xtier-env/bin/python ] && [ -f /data/src/xtier_86kd/deploy/tf500_api_proxy.py ]; do
  sleep 2
done

TARGET_HOST=${TF500_HOST:-}
if [ -z "$TARGET_HOST" ] && [ -r /data/xtier-86kd-console/bind-host ]; then
  TARGET_HOST=$(tr -d ' \t\r\n' < /data/xtier-86kd-console/bind-host)
fi
if [ -n "$TARGET_HOST" ]; then
  until ip -o -4 addr show | grep -Fq "${TARGET_HOST}/"; do
    sleep 2
  done
else
  until ip -o -4 addr show dev tailscale0 2>/dev/null | grep -Eq 'inet [0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/'; do
    sleep 2
  done
fi

mkdir -p /data/xtier-86kd-console
exec >> /data/xtier-86kd-console/proxy.log 2>&1
exec /usr/bin/flock -n /data/xtier-86kd-console/proxy.lock \
  /data/xtier-env/bin/python -u /data/src/xtier_86kd/deploy/tf500_api_proxy.py
