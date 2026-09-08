# SPDX-FileCopyrightText: 2026 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Caller contract: every Backend-bound internal API call made by
executor_manager must carry the internal service token as a Bearer header.

These tests capture the real outbound request headers at the actual call
sites (not a shared header helper) so a regression that drops the header at
any call site fails here."""

from types import SimpleNamespace

import pytest

from executor_manager.clients.callback_client import CallbackClient
from executor_manager.routers import routers
from executor_manager.services.sandbox.manager import SandboxManager

INTERNAL_TOKEN = "test-internal-token"


class _RecordingPostClient:
    """httpx/traced client double that records url, json and headers."""

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


@pytest.mark.asyncio
async def test_callback_client_sends_bearer_header(mocker, monkeypatch):
    monkeypatch.setattr(
        "executor_manager.clients.callback_client.INTERNAL_SERVICE_TOKEN",
        INTERNAL_TOKEN,
    )
    requests = []
    mocker.patch(
        "executor_manager.clients.callback_client.traced_async_client",
        return_value=_RecordingClientContext(requests),
    )

    client = CallbackClient()
    result = await client.send_error(task_id=7, subtask_id=1, error_message="boom")

    assert result is True
    assert len(requests) == 1
    url, _, headers = requests[0]
    assert url.endswith("/api/internal/callback")
    assert headers.get("Authorization") == f"Bearer {INTERNAL_TOKEN}"


@pytest.mark.asyncio
async def test_callback_proxy_forwards_bearer_header(mocker, monkeypatch):
    monkeypatch.setattr("executor_manager.routers.routers.INTERNAL_SERVICE_TOKEN", INTERNAL_TOKEN)
    requests = []
    mocker.patch(
        "executor_manager.routers.routers.traced_async_client",
        return_value=_RecordingClientContext(requests),
    )
    mocker.patch("executor_manager.routers.routers.set_task_context")
    mocker.patch("executor_manager.routers.routers._update_validation_status_from_callback")
    tracker = mocker.Mock()
    mocker.patch(
        "executor_manager.services.task_heartbeat_manager.get_running_task_tracker",
        return_value=tracker,
    )

    event_data = {
        "event_type": "response.output_text.delta",
        "task_id": 42,
        "subtask_id": 1,
        "delta": "hello",
    }
    http_request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))
    await routers.callback_handler(event_data, http_request)

    assert len(requests) == 1
    url, _, headers = requests[0]
    assert url.endswith("/api/internal/callback")
    assert headers.get("Authorization") == f"Bearer {INTERNAL_TOKEN}"


@pytest.mark.asyncio
async def test_sandbox_archive_callback_sends_bearer_header(mocker, monkeypatch):
    monkeypatch.setattr(
        "executor_manager.services.sandbox.manager.INTERNAL_SERVICE_TOKEN",
        INTERNAL_TOKEN,
    )
    requests = []
    fake_client = _RecordingClientContext(requests)
    mocker.patch("httpx.AsyncClient", return_value=fake_client)

    manager = SandboxManager.__new__(SandboxManager)
    sandbox = SimpleNamespace(
        sandbox_id="sandbox-1",
        metadata={"task_id": 9},
        executor_name="executor-9",
        executor_namespace="default",
    )
    result = await manager._post_workspace_archive_callback(
        url="http://backend:8000/api/internal/workspace-archives/9/restore-sandbox",
        payload={"executor_name": "executor-9"},
        action="restore",
        sandbox=sandbox,
    )

    assert result is True
    assert len(requests) == 1
    url, _, headers = requests[0]
    assert url.endswith("/api/internal/workspace-archives/9/restore-sandbox")
    assert headers.get("Authorization") == f"Bearer {INTERNAL_TOKEN}"


@pytest.mark.asyncio
async def test_callers_omit_bearer_header_when_token_unconfigured(mocker, monkeypatch):
    """Without a configured token the header is omitted rather than sent
    empty, so the call site stays on the network path it had before."""
    monkeypatch.setattr(
        "executor_manager.clients.callback_client.INTERNAL_SERVICE_TOKEN", ""
    )
    requests = []
    mocker.patch(
        "executor_manager.clients.callback_client.traced_async_client",
        return_value=_RecordingClientContext(requests),
    )

    client = CallbackClient()
    await client.send_error(task_id=7, subtask_id=1, error_message="boom")

    assert len(requests) == 1
    _, _, headers = requests[0]
    assert headers.get("Authorization") is None
