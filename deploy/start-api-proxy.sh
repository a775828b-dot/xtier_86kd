#!/usr/bin/env bash
set -euo pipefail

until [ -x /data/xtier-env/bin/python ] && [ -f /data/src/xtier_86kd/deploy/tf500_api_proxy.py ]; do
  sleep 2
done

until ip -o -4 addr show | grep -Fq '<TF500_IP>/'; do
  sleep 2
done

mkdir -p /data/xtier-86kd-console
exec >> /data/xtier-86kd-console/proxy.log 2>&1
exec /usr/bin/flock -n /data/xtier-86kd-console/proxy.lock \
  /data/xtier-env/bin/python -u /data/src/xtier_86kd/deploy/tf500_api_proxy.py
