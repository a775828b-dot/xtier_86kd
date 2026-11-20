"""Prefix-sticky Data-Parallel (DP) router for TF500 API proxy (P0-N1).

Routes multi-turn Agent requests with shared conversation prefixes to the same
SGLang DP rank so per-rank RadixCache stays hot, while preventing shared
system-prompt hotspots and rank overload.

Environment variables:
  TF500_PROXY_STICKY_DP          "1" (default) enables sticky DP routing; "0" disables.
  TF500_PROXY_DP_SIZE            Number of DP ranks (default "8"). <= 1 disables routing.
  TF500_PROXY_STICKY_MAX_ENTRIES Max prefix-hash LRU entries (default "100000").
  TF500_PROXY_STICKY_TTL_S       LRU entry TTL in seconds (default "7200").
  TF500_PROXY_STICKY_MAX_INFLIGHT Hard inflight request cap per rank before fallback (default "8").
  TF500_PROXY_STICKY_MAX_SKEW    Max inflight skew vs least-loaded rank before fallback (default "4").
"""

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import json
import os
import threading
import time

ROUTABLE_PATHS = {
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
}
HEADER_DP_RANK = "X-Data-Parallel-Rank"
HEADER_OVERRIDE_DP_RANK = "x-override-routed-dp-rank"
SYSTEM_ROLES = {"system", "developer"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in ("0", "false", "off", "no")


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        return default
    return max(minimum, value)


def _canonical_bytes(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _step_hash(prev: bytes, payload: bytes) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(prev)
    h.update(b"\x00")
    h.update(payload)
    return h.digest()


def extract_prefix_hashes(payload: dict, path: str) -> list[bytes]:
    """Compute cumulative prefix digests eligible for sticky DP lookup.

    For chat completions, pure system/developer prefixes are intentionally
    excluded from the returned list so concurrent agents sharing the same
    system prompt + tools schema do not hotspot onto a single DP rank.
    """
    if not isinstance(payload, dict):
        return []

    messages = payload.get("messages")
    if isinstance(messages, list) and messages:
        tools = payload.get("tools")
        digest = _step_hash(b"chat:v1", _canonical_bytes(tools) if tools is not None else b"")
        eligible: list[bytes] = []
        seen_non_system = False
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            norm = {
                "role": role,
                "content": msg.get("content"),
            }
            if msg.get("name") is not None:
                norm["name"] = msg.get("name")
            if msg.get("tool_call_id") is not None:
                norm["tool_call_id"] = msg.get("tool_call_id")
            if msg.get("tool_calls") is not None:
                norm["tool_calls"] = msg.get("tool_calls")
            digest = _step_hash(digest, _canonical_bytes(norm))
            if role not in SYSTEM_ROLES:
                seen_non_system = True
            if seen_non_system:
                eligible.append(digest)
        return eligible

    # /v1/completions or /v1/responses fallback
    prompt = payload.get("prompt")
    if prompt is None and path == "/v1/responses":
        prompt = payload.get("input")

    if isinstance(prompt, str) and len(prompt) >= 256:
        step = 4096
        digest = b"prompt:str:v1"
        eligible = []
        n = len(prompt)
        for start in range(0, n, step):
            chunk = prompt[start:start + step].encode("utf-8", "replace")
            digest = _step_hash(digest, chunk)
            eligible.append(digest)
        return eligible

    if isinstance(prompt, list) and prompt and all(isinstance(x, int) for x in prompt[:16]):
        if len(prompt) < 64:
            return []
        step = 1024
        digest = b"prompt:ids:v1"
        eligible = []
        for start in range(0, len(prompt), step):
            chunk = _canonical_bytes(prompt[start:start + step])
            digest = _step_hash(digest, chunk)
            eligible.append(digest)
        return eligible

    return []


def extract_client_dp_rank(headers: dict, payload: dict | None, dp_size: int) -> int | None:
    """Return explicit client-requested DP rank if valid, else None."""
    if dp_size <= 0:
        return None
    if headers:
        for name, value in headers.items():
            low = name.lower()
            if low in ("x-data-parallel-rank", "x-override-routed-dp-rank"):
                try:
                    rank = int(str(value).strip())
                except (TypeError, ValueError):
                    continue
                if 0 <= rank < dp_size:
                    return rank
    if isinstance(payload, dict) and payload.get("routed_dp_rank") is not None:
        try:
            rank = int(payload["routed_dp_rank"])
        except (TypeError, ValueError):
            return None
        if 0 <= rank < dp_size:
            return rank
    return None


@dataclass
class RouteLease:
    rank: int | None
    reason: str
    inject_headers: bool = False
    hashes: list[bytes] = field(default_factory=list)
    est_tokens: int = 0
    tracked: bool = False
    released: bool = False


class PrefixStickyDPRouter:
    """Thread-safe prefix-sticky DP router with anti-hotspot & overload guard."""

    def __init__(
        self,
        dp_size: int | None = None,
        max_entries: int | None = None,
        ttl_s: float | None = None,
        max_inflight: int | None = None,
        max_skew: int | None = None,
        enabled: bool | None = None,
        clock=time.monotonic,
    ):
        self._explicit_enabled = enabled
        self._explicit_dp_size = dp_size
        self.dp_size = dp_size if dp_size is not None else _env_int("TF500_PROXY_DP_SIZE", 8, 1)
        self.max_entries = max_entries if max_entries is not None else _env_int("TF500_PROXY_STICKY_MAX_ENTRIES", 100000, 16)
        self.ttl_s = float(ttl_s if ttl_s is not None else _env_int("TF500_PROXY_STICKY_TTL_S", 7200, 1))
        self.max_inflight = max_inflight if max_inflight is not None else _env_int("TF500_PROXY_STICKY_MAX_INFLIGHT", 8, 1)
        self.max_skew = max_skew if max_skew is not None else _env_int("TF500_PROXY_STICKY_MAX_SKEW", 4, 1)
        self._clock = clock

        self._lock = threading.Lock()
        self._lru: OrderedDict[bytes, tuple[int, float]] = OrderedDict()
        self._inflight_reqs: list[int] = [0] * self.dp_size
        self._inflight_tokens: list[int] = [0] * self.dp_size
        self._rr_cursor: int = 0

        self._hits = 0
        self._misses = 0
        self._overload_fallbacks = 0
        self._client_overrides = 0
        self._disabled_count = 0

    def is_enabled(self) -> bool:
        if self._explicit_enabled is not None:
            return bool(self._explicit_enabled) and self.dp_size > 1
        return _env_bool("TF500_PROXY_STICKY_DP", True) and self.dp_size > 1

    def reset(self) -> None:
        with self._lock:
            self._lru.clear()
            self._inflight_reqs = [0] * self.dp_size
            self._inflight_tokens = [0] * self.dp_size
            self._rr_cursor = 0
            self._hits = 0
            self._misses = 0
            self._overload_fallbacks = 0
            self._client_overrides = 0
            self._disabled_count = 0

    def _pick_least_loaded_locked(self) -> int:
        n = self.dp_size
        cursor = self._rr_cursor
        best_rank = cursor
        best_key = (self._inflight_reqs[cursor], self._inflight_tokens[cursor], 0)
        for offset in range(1, n):
            r = (cursor + offset) % n
            key = (self._inflight_reqs[r], self._inflight_tokens[r], offset)
            if key < best_key:
                best_key = key
                best_rank = r
        self._rr_cursor = (best_rank + 1) % n
        return best_rank

    def _lookup_longest_locked(self, hashes: list[bytes], now: float) -> int | None:
        for h in reversed(hashes):
            entry = self._lru.get(h)
            if entry is None:
                continue
            rank, ts = entry
            if now - ts > self.ttl_s:
                del self._lru[h]
                continue
            if 0 <= rank < self.dp_size:
                self._lru.move_to_end(h)
                return rank
        return None

    def _record_hashes_locked(self, hashes: list[bytes], rank: int, now: float) -> None:
        if not hashes or not (0 <= rank < self.dp_size):
            return
        for h in hashes:
            self._lru[h] = (rank, now)
            self._lru.move_to_end(h)
        while len(self._lru) > self.max_entries:
            self._lru.popitem(last=False)

    def acquire(
        self,
        path: str,
        headers: dict,
        payload: dict | None,
        raw_len: int = 0,
    ) -> RouteLease:
        if path not in ROUTABLE_PATHS:
            return RouteLease(rank=None, reason="passthrough")

        est_tokens = max(1, int(raw_len) // 4) if raw_len > 0 else 1
        hashes = extract_prefix_hashes(payload, path) if isinstance(payload, dict) else []

        client_rank = extract_client_dp_rank(headers, payload, self.dp_size)
        if client_rank is not None:
            with self._lock:
                self._inflight_reqs[client_rank] += 1
                self._inflight_tokens[client_rank] += est_tokens
                self._client_overrides += 1
            return RouteLease(
                rank=client_rank,
                reason="client_override",
                inject_headers=False,
                hashes=hashes,
                est_tokens=est_tokens,
                tracked=True,
            )

        if not self.is_enabled():
            with self._lock:
                self._disabled_count += 1
            return RouteLease(rank=None, reason="disabled")

        now = self._clock()
        with self._lock:
            hit_rank = self._lookup_longest_locked(hashes, now)
            if hit_rank is not None:
                min_reqs = min(self._inflight_reqs)
                overloaded = (
                    self._inflight_reqs[hit_rank] >= self.max_inflight
                    or (self._inflight_reqs[hit_rank] - min_reqs) >= self.max_skew
                )
                if overloaded:
                    rank = self._pick_least_loaded_locked()
                    reason = "overload_fallback"
                    self._overload_fallbacks += 1
                else:
                    rank = hit_rank
                    reason = "sticky_hit"
                    self._hits += 1
            else:
                rank = self._pick_least_loaded_locked()
                reason = "least_loaded"
                self._misses += 1

            self._inflight_reqs[rank] += 1
            self._inflight_tokens[rank] += est_tokens

        return RouteLease(
            rank=rank,
            reason=reason,
            inject_headers=True,
            hashes=hashes,
            est_tokens=est_tokens,
            tracked=True,
        )

    def release(self, lease: RouteLease | None, ok: bool = True) -> None:
        if lease is None or lease.released or not lease.tracked or lease.rank is None:
            return
        lease.released = True
        now = self._clock()
        with self._lock:
            r = lease.rank
            if 0 <= r < self.dp_size:
                self._inflight_reqs[r] = max(0, self._inflight_reqs[r] - 1)
                self._inflight_tokens[r] = max(0, self._inflight_tokens[r] - lease.est_tokens)
                if ok and lease.hashes:
                    self._record_hashes_locked(lease.hashes, r, now)

    def stats(self) -> dict:
        with self._lock:
            routed = self._hits + self._misses + self._overload_fallbacks
            hit_rate = round(100.0 * self._hits / routed, 1) if routed > 0 else 0.0
            return {
                "enabled": self.is_enabled(),
                "dp_size": self.dp_size,
                "lru_size": len(self._lru),
                "inflight_reqs": list(self._inflight_reqs),
                "hits": self._hits,
                "misses": self._misses,
                "overload_fallbacks": self._overload_fallbacks,
                "client_overrides": self._client_overrides,
                "disabled_count": self._disabled_count,
                "sticky_hit_rate_pct": hit_rate,
            }