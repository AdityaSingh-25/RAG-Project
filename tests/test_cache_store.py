import time
from pathlib import Path

from rag_engine.cache.store import CacheStore


def _store(tmp_path: Path, ttl: int = 3600) -> CacheStore:
    return CacheStore(path=tmp_path / "cache.sqlite", default_ttl_seconds=ttl)


def test_set_and_get_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.set("ns", "k", {"a": 1, "b": [2, 3]})
    assert store.get("ns", "k") == {"a": 1, "b": [2, 3]}


def test_missing_key_returns_none(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.get("ns", "absent") is None


def test_namespaces_are_isolated(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.set("a", "k", 1)
    store.set("b", "k", 2)
    assert store.get("a", "k") == 1
    assert store.get("b", "k") == 2


def test_expired_value_returns_none(tmp_path: Path) -> None:
    store = _store(tmp_path, ttl=3600)
    store.set("ns", "k", "v", ttl_seconds=0)
    # An immediate get should observe expires_at <= now and treat as miss.
    time.sleep(0.01)
    assert store.get("ns", "k") is None


def test_clear_namespace_only(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.set("a", "k", 1)
    store.set("b", "k", 2)
    removed = store.clear("a")
    assert removed == 1
    assert store.get("a", "k") is None
    assert store.get("b", "k") == 2


def test_purge_expired_removes_only_stale_rows(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.set("ns", "fresh", 1, ttl_seconds=3600)
    store.set("ns", "stale", 2, ttl_seconds=0)
    time.sleep(0.01)
    removed = store.purge_expired()
    assert removed == 1
    assert store.get("ns", "fresh") == 1
    assert store.get("ns", "stale") is None


def test_tuples_are_serialized_as_lists(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.set("ns", "k", ("a", "b"))
    assert store.get("ns", "k") == ["a", "b"]
