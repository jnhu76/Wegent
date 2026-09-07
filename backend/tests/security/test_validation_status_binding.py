# SPDX-FileCopyrightText: 2026 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Validation-status callback security contract.

Boundary: the callback endpoint requires internal service authentication.
Binding: state may never be created from a caller-supplied validation_id, and
container cleanup may only target the executor identity already bound to a
trusted validation record created by POST /shells/validate-image.
"""

import asyncio
from typing import Any, Optional

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api.endpoints.adapter import shells as shells_module
from app.api.endpoints.adapter.shells import (
    ValidationStatusUpdateRequest,
    update_validation_status,
)
from app.core.config import settings

INTERNAL_TOKEN = "test-internal-token"

VALIDATION_URL = "/api/shells/validation-status/val-123"

pytestmark = pytest.mark.asyncio


class FakeCache:
    """Minimal async cache double; records writes so tests can assert on them."""

    def __init__(self, initial: Optional[dict[str, Any]] = None) -> None:
        self.store: dict[str, Any] = {}
        self.writes: list[str] = []
        if initial is not None:
            self.store[self._key("val-123")] = dict(initial)

    @staticmethod
    def _key(validation_id: str) -> str:
        return f"{shells_module.VALIDATION_STATUS_KEY_PREFIX}{validation_id}"

    async def get(self, key: str) -> Any:
        return self.store.get(key)

    async def get_strict(self, key: str) -> Any:
        return self.store.get(key)

    async def set(self, key: str, value: Any, expire: Optional[int] = None) -> None:
        self.writes.append(key)
        self.store[key] = value


@pytest.fixture(autouse=True)
def configure_internal_service_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "INTERNAL_SERVICE_TOKEN", INTERNAL_TOKEN)


@pytest.fixture
def fake_cache(monkeypatch: pytest.MonkeyPatch) -> FakeCache:
    cache = FakeCache()
    monkeypatch.setattr(shells_module, "cache_manager", cache)
    return cache


def seeded_record() -> dict[str, Any]:
    """A validation record exactly as POST /shells/validate-image creates it."""
    return {
        "validation_id": "val-123",
        "status": "submitted",
        "stage": "Validation task submitted",
        "progress": 10,
        "valid": None,
        "checks": None,
        "errors": None,
        "error_message": None,
    }


def test_anonymous_callback_rejected_without_side_effects(
    test_client: TestClient, fake_cache: FakeCache
) -> None:
    response = test_client.post(
        VALIDATION_URL,
        json={"status": "completed", "progress": 100, "valid": True},
    )

    assert response.status_code == 401
    assert fake_cache.writes == []


def test_invalid_token_callback_rejected(test_client: TestClient) -> None:
    response = test_client.post(
        VALIDATION_URL,
        json={"status": "completed", "valid": True},
        headers={"Authorization": "Bearer wrong-token"},
    )

    assert response.status_code == 401


def test_unknown_validation_id_is_rejected_and_creates_no_state(
    test_client: TestClient, fake_cache: FakeCache
) -> None:
    response = test_client.post(
        "/api/shells/validation-status/attacker-chosen-id",
        json={
            "status": "completed",
            "progress": 100,
            "valid": True,
            "executor_name": "victim-container",
        },
        headers={"Authorization": f"Bearer {INTERNAL_TOKEN}"},
    )

    assert response.status_code == 404
    assert fake_cache.writes == []
    assert "attacker-chosen-id" not in fake_cache.store


class FailingCache(FakeCache):
    """Cache double whose backend is down: reads raise instead of returning a
    miss, mirroring what get_strict propagates."""

    async def get_strict(self, key: str) -> Any:
        raise RuntimeError("cache unavailable")


def test_cache_outage_returns_503_not_false_404(
    test_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The executor deletes containers on 404, so a transient store outage
    must surface as 503 — never as a missing-record 404 that would reap a
    live validation."""
    cache = FailingCache()
    monkeypatch.setattr(shells_module, "cache_manager", cache)

    response = test_client.post(
        VALIDATION_URL,
        json={"status": "running", "progress": 50},
        headers={"Authorization": f"Bearer {INTERNAL_TOKEN}"},
    )

    assert response.status_code == 503
    assert cache.writes == []


async def test_mismatched_executor_name_rejected_without_cleanup(
    monkeypatch: pytest.MonkeyPatch, fake_cache: FakeCache
) -> None:
    fake_cache.store[fake_cache._key("val-123")] = seeded_record() | {
        "executor_name": "wegent-executor-trusted"
    }
    cleanup_calls: list[str] = []

    async def fake_cleanup(executor_name: str) -> None:
        cleanup_calls.append(executor_name)

    monkeypatch.setattr(shells_module, "_cleanup_validation_container", fake_cleanup)

    with pytest.raises(HTTPException) as exc_info:
        await update_validation_status(
            "val-123",
            ValidationStatusUpdateRequest(
                status="completed",
                progress=100,
                valid=True,
                executor_name="attacker-container",
            ),
        )

    assert exc_info.value.status_code == 403
    assert cleanup_calls == []
    # The rejected report must not mutate the trusted record.
    assert fake_cache.store[fake_cache._key("val-123")]["executor_name"] == (
        "wegent-executor-trusted"
    )
    assert fake_cache.store[fake_cache._key("val-123")]["valid"] is None


async def test_matching_executor_completion_triggers_cleanup(
    monkeypatch: pytest.MonkeyPatch, fake_cache: FakeCache
) -> None:
    fake_cache.store[fake_cache._key("val-123")] = seeded_record() | {
        "executor_name": "wegent-executor-trusted"
    }
    cleanup_calls: list[str] = []

    async def fake_cleanup(executor_name: str) -> None:
        cleanup_calls.append(executor_name)

    monkeypatch.setattr(shells_module, "_cleanup_validation_container", fake_cleanup)

    await update_validation_status(
        "val-123",
        ValidationStatusUpdateRequest(
            status="completed",
            progress=100,
            valid=True,
            executor_name="wegent-executor-trusted",
        ),
    )
    # Yield once so the scheduled cleanup task runs.
    await asyncio.sleep(0)

    assert cleanup_calls == ["wegent-executor-trusted"]
    record = fake_cache.store[fake_cache._key("val-123")]
    assert record["status"] == "completed"
    assert record["valid"] is True


async def test_first_executor_report_binds_identity(
    monkeypatch: pytest.MonkeyPatch, fake_cache: FakeCache
) -> None:
    """Legitimate flow: the completion callback is the first carrier of the
    executor name; it is bound to the trusted record and honored."""
    fake_cache.store[fake_cache._key("val-123")] = seeded_record()
    cleanup_calls: list[str] = []

    async def fake_cleanup(executor_name: str) -> None:
        cleanup_calls.append(executor_name)

    monkeypatch.setattr(shells_module, "_cleanup_validation_container", fake_cleanup)

    await update_validation_status(
        "val-123",
        ValidationStatusUpdateRequest(
            status="completed",
            progress=100,
            valid=True,
            executor_name="wegent-executor-abc123",
        ),
    )
    await asyncio.sleep(0)

    assert cleanup_calls == ["wegent-executor-abc123"]
    assert (
        fake_cache.store[fake_cache._key("val-123")]["executor_name"]
        == "wegent-executor-abc123"
    )


async def test_stage_update_without_executor_name_still_works(
    monkeypatch: pytest.MonkeyPatch, fake_cache: FakeCache
) -> None:
    fake_cache.store[fake_cache._key("val-123")] = seeded_record()

    await update_validation_status(
        "val-123",
        ValidationStatusUpdateRequest(
            status="running",
            stage="pulling_image",
            progress=30,
        ),
    )

    record = fake_cache.store[fake_cache._key("val-123")]
    assert record["status"] == "running"
    assert record["progress"] == 30
