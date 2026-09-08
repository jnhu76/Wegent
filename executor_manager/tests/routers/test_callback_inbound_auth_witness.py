# SPDX-FileCopyrightText: 2026 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Known security witness: unauthenticated callback proxy credential delegation.

BUG-002 closes the Backend direct internal-auth gap by attaching
``verify_internal_service_token`` to Backend ``/api/internal/callback``.

However, executor-manager's inbound ``POST /executor-manager/callback`` remains
unauthenticated and now attaches the internal service token when forwarding to
Backend. An anonymous caller able to reach :8001 can therefore ask
executor-manager to act as an authenticated deputy (confused-deputy problem):
the outbound backend request carries ``Authorization: Bearer <token>`` that the
anonymous caller never possessed.

This test asserts the secure invariant -- an anonymous callback must be
rejected before any outbound request -- and is marked ``xfail(strict=True)``
because the invariant is currently violated. It is intentionally red and must
stay red until a maintainer defines the executor callback trust model. Do not
remove the xfail marker or convert this into a passing characterization of the
current behavior; that would be read as accepting the delegation.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

TOKEN = "secret-internal-token"
TASK_API_DOMAIN = "http://backend.internal:8000"

ANONYMOUS_EVENT = {
    "event_type": "response.output_text.delta",
    "task_id": 42,
    "subtask_id": 1,
    "delta": "hello from anonymous caller",
}


class _RecordingPostClient:
    """httpx client double that records the actual outbound request."""

    def __init__(self, requests):
        self.requests = requests

    async def post(self, url, json=None, headers=None):
        self.requests.append((url, json, headers))
        return SimpleNamespace(
            status_code=200,
            text="ok",
            json=lambda: {"success": True},
            raise_for_status=lambda: None,
        )


class _RecordingClientContext:
    def __init__(self, requests):
        self.requests = requests

    async def __aenter__(self):
        return _RecordingPostClient(self.requests)

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def app():
    with patch(
        "executor_manager.executors.docker.executor.subprocess.run",
        return_value=SimpleNamespace(stdout="Docker version 27.0.0", returncode=0),
    ):
        from executor_manager.routers.routers import app as router_app

        return router_app


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known BUG-002 closure blocker: unauthenticated executor-manager callback "
        "proxy delegates backend internal-service credentials. Requires maintainer "
        "decision on the executor callback trust model."
    ),
)
def test_anonymous_callback_cannot_delegate_backend_credentials(app, monkeypatch):
    """Secure invariant: an anonymous callback must be rejected before any
    outbound backend request, so no internal-service credential is delegated.

    Today this fails: the handler forwards with
    ``Authorization: Bearer secret-internal-token`` to
    ``http://backend.internal:8000/api/internal/callback`` and returns 200.
    The failure message records the exact observed delegation.
    """
    requests = []
    monkeypatch.setattr(
        "executor_manager.routers.routers.INTERNAL_SERVICE_TOKEN", TOKEN
    )
    monkeypatch.setenv("TASK_API_DOMAIN", TASK_API_DOMAIN)
    monkeypatch.setattr(
        "executor_manager.routers.routers.traced_async_client",
        lambda **kwargs: _RecordingClientContext(requests),
    )
    monkeypatch.setattr(
        "executor_manager.routers.routers.set_task_context",
        lambda **kwargs: None,
    )

    # Anonymous inbound: no Authorization header is supplied.
    response = TestClient(app).post("/executor-manager/callback", json=ANONYMOUS_EVENT)

    assert response.status_code in (401, 403), (
        f"WITNESS: anonymous callback accepted with HTTP {response.status_code}; "
        f"outbound delegation observed: {requests}"
    )
    assert len(requests) == 0, (
        f"WITNESS: outbound backend request issued for anonymous callback: {requests}"
    )
