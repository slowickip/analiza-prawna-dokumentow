from __future__ import annotations

from uuid import uuid4

import pytest

from contract_analyzer.config import RunConfig
from contract_analyzer.storage import ContentExpired, RunTextStore


@pytest.fixture
def store() -> RunTextStore:
    return RunTextStore()


def test_put_and_get_round_trip(store: RunTextStore) -> None:
    key = store.put(b"canonical text")
    assert store.get(key) == b"canonical text"
    assert store.contains(key) is True


def test_contains_unknown_key_returns_false(store: RunTextStore) -> None:
    assert store.contains(uuid4()) is False


def test_contains_purges_expired_key() -> None:
    now = {"value": 0.0}
    store = RunTextStore(
        RunConfig(content_ttl_seconds=10),
        clock=lambda: now["value"],
    )
    key = store.put(b"ephemeral")
    assert store.contains(key) is True
    now["value"] = 11.0
    assert store.contains(key) is False


def test_get_unknown_key_raises_content_expired(store: RunTextStore) -> None:
    with pytest.raises(ContentExpired):
        store.get(uuid4())


def test_delete_prevents_subsequent_get(store: RunTextStore) -> None:
    key = store.put(b"to delete")
    store.delete(key)
    with pytest.raises(ContentExpired):
        store.get(key)


def test_expire_prevents_subsequent_get(store: RunTextStore) -> None:
    key = store.put(b"to expire")
    store.expire(key)
    with pytest.raises(ContentExpired):
        store.get(key)


def test_delete_unknown_key_is_noop(store: RunTextStore) -> None:
    store.delete(uuid4())


def test_expire_unknown_key_is_noop() -> None:
    RunTextStore().expire(uuid4())


def test_ttl_does_not_expire_before_deadline() -> None:
    now = {"value": 0.0}
    store = RunTextStore(
        RunConfig(content_ttl_seconds=60),
        clock=lambda: now["value"],
    )
    key = store.put(b"still valid")
    now["value"] = 59.0
    assert store.get(key) == b"still valid"


def test_ttl_expiry_purges_on_get() -> None:
    now = {"value": 0.0}
    store = RunTextStore(
        RunConfig(content_ttl_seconds=10),
        clock=lambda: now["value"],
    )
    key = store.put(b"expires soon")
    now["value"] = 11.0
    with pytest.raises(ContentExpired):
        store.get(key)
    with pytest.raises(ContentExpired):
        store.get(key)


def test_keys_are_unique(store: RunTextStore) -> None:
    first = store.put(b"a")
    second = store.put(b"b")
    assert first != second


def test_configured_ttl_governs_expiry() -> None:
    now = {"value": 0.0}
    store = RunTextStore(
        RunConfig(content_ttl_seconds=1),
        clock=lambda: now["value"],
    )
    key = store.put(b"short lived")
    now["value"] = 2.0
    assert store.purge_expired() == 1
    with pytest.raises(ContentExpired):
        store.get(key)


def test_purge_expired_sweeps_all_overdue_keys_without_get() -> None:
    now = {"value": 0.0}
    store = RunTextStore(
        RunConfig(content_ttl_seconds=10),
        clock=lambda: now["value"],
    )
    expired_key = store.put(b"should expire")
    backing, _ = store._store[expired_key]
    now["value"] = 11.0
    store.put(b"still fresh")
    assert store.purge_expired() == 1
    assert all(byte == 0 for byte in backing)
    with pytest.raises(ContentExpired):
        store.get(expired_key)


def test_delete_zeroes_backing_bytes() -> None:
    store = RunTextStore()
    key = store.put(b"secret")
    backing, _ = store._store[key]
    store.delete(key)
    assert all(byte == 0 for byte in backing)


def test_expire_zeroes_backing_bytes() -> None:
    store = RunTextStore()
    key = store.put(b"secret")
    backing, _ = store._store[key]
    store.expire(key)
    assert all(byte == 0 for byte in backing)
