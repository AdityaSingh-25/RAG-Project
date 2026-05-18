from collections.abc import Iterable
from typing import Any


def filter_records(records: Iterable[dict[str, Any]], filters: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for record in records:
        if all(record.get(key) == value for key, value in filters.items()):
            results.append(record)
    return results

