"""Ephemeral document text storage."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from uuid import UUID, uuid4

from contract_analyzer.config import RunConfig


class ContentExpired(Exception):
    """Raised when ephemeral content was deleted or expired."""


class RunTextStore:
    """Locked in-process store for ephemeral document session content."""

    def __init__(
        self,
        config: RunConfig | None = None,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        resolved = config or RunConfig()
        self._ttl_seconds = resolved.content_ttl_seconds
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._store: dict[UUID, tuple[bytearray, float]] = {}

    def contains(self, key: UUID) -> bool:
        with self._lock:
            self._purge_if_expired_locked(key)
            return key in self._store

    def put(self, content: bytes) -> UUID:
        key = uuid4()
        with self._lock:
            self._store[key] = (
                bytearray(content),
                self._clock() + self._ttl_seconds,
            )
        return key

    def get(self, key: UUID) -> bytes:
        with self._lock:
            self._purge_if_expired_locked(key)
            entry = self._store.get(key)
            if entry is None:
                raise ContentExpired(f"content expired for {key}")
            return bytes(entry[0])

    def delete(self, key: UUID) -> None:
        with self._lock:
            self._purge_key_locked(key)

    def expire(self, key: UUID) -> None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return
            data, _ = entry
            self._store[key] = (data, self._clock())
            self._purge_if_expired_locked(key)

    def purge_expired(self) -> int:
        with self._lock:
            now = self._clock()
            overdue = [
                key for key, (_, deadline) in self._store.items() if now >= deadline
            ]
            for key in overdue:
                self._purge_key_locked(key)
            return len(overdue)

    def _purge_key_locked(self, key: UUID) -> None:
        entry = self._store.pop(key, None)
        if entry is not None:
            entry[0][:] = b"\x00" * len(entry[0])

    def _purge_if_expired_locked(self, key: UUID) -> None:
        entry = self._store.get(key)
        if entry is None:
            return
        if self._clock() >= entry[1]:
            self._purge_key_locked(key)
