"""Dynamic bind/proxy host resolution for TF500 without hardcoding private IPs.

Resolution priority:
1. Environment variable ``TF500_HOST``
2. Local state file ``/data/xtier-86kd-console/bind-host`` (untracked)
3. Linux ``tailscale0`` interface IPv4 address (``ip -o -4 addr show dev tailscale0``)
4. ``tailscale ip -4`` CLI output
5. Fallback ``default`` (``127.0.0.1``)
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

HOST_STATE_FILE = Path("/data/xtier-86kd-console/bind-host")
_IPV4_CIDR_RE = re.compile(r"inet\s+(\d+\.\d+\.\d+\.\d+)/\d+")
_IPV4_EXACT_RE = re.compile(r"\d+\.\d+\.\d+\.\d+")


def _parse_tailscale_ip(text: str) -> str | None:
    m = _IPV4_CIDR_RE.search(text)
    if m:
        return m.group(1)
    for line in text.splitlines():
        cand = line.strip()
        if _IPV4_EXACT_RE.fullmatch(cand):
            return cand
    return None


def resolve_host(default: str = "127.0.0.1") -> str:
    env_val = os.environ.get("TF500_HOST", "").strip()
    if env_val:
        return env_val
    try:
        if HOST_STATE_FILE.is_file():
            file_val = HOST_STATE_FILE.read_text(encoding="utf-8").strip()
            if file_val:
                return file_val
    except OSError:
        pass
    for cmd in (
        ["ip", "-o", "-4", "addr", "show", "dev", "tailscale0"],
        ["tailscale", "ip", "-4"],
    ):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=2, check=False)
            if proc.returncode == 0 and proc.stdout:
                detected = _parse_tailscale_ip(proc.stdout)
                if detected:
                    return detected
        except (OSError, subprocess.SubprocessError):
            continue
    return default
