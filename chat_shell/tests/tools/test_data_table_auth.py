# SPDX-FileCopyrightText: 2026 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for DataTableTool backend internal API authentication.

Regression witness for the BUG-002 caller-sync contract: the tool must use
the canonical ``settings.backend_internal_token`` resolver
(``REMOTE_STORAGE_TOKEN or INTERNAL_SERVICE_TOKEN``) so that a deployment
configured with only ``REMOTE_STORAGE_TOKEN`` still authenticates to the
protected ``/api/internal/tables/query`` endpoint.
"""

import json

import pytest
from pytest_httpx import HTTPXMock

from chat_shell.core.config import settings
from chat_shell.tools.builtin import DataTableTool

BACKEND_RESPONSE = {
    "schema": [{"field_name": "id", "field_type": "number"}],
    "records": [{"id": 1}],
    "total_count": 1,
}


def _make_tool() -> DataTableTool:
    return DataTableTool(
        table_contexts=[
            {
                "name": "test-table",
                "baseId": "base-1",
                "sheetIdOrName": "sheet-1",
            }
        ],
        user_id=42,
        user_name="tester",
    )


async def _query(tool: DataTableTool) -> dict:
    result = await tool._arun(
        provider="dingtalk",
        base_id="base-1",
        sheet_id_or_name="sheet-1",
        max_records=100,
    )
    return json.loads(result)


@pytest.mark.asyncio
async def test_query_uses_remote_storage_token_when_configured_alone(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REMOTE_STORAGE_TOKEN-only deployment must still authenticate (BLOCKER-2 witness)."""
    monkeypatch.setattr(
        settings, "REMOTE_STORAGE_URL", "http://backend.internal:8000/api/internal"
    )
    monkeypatch.setattr(settings, "REMOTE_STORAGE_TOKEN", "remote-token")
    monkeypatch.setattr(settings, "INTERNAL_SERVICE_TOKEN", "")
    httpx_mock.add_response(
        url="http://backend.internal:8000/api/internal/tables/query",
        json=BACKEND_RESPONSE,
    )

    result = await _query(_make_tool())

    request = httpx_mock.get_request()
    assert result["total_count"] == 1
    assert request.headers["Authorization"] == "Bearer remote-token"


@pytest.mark.asyncio
async def test_query_falls_back_to_internal_service_token(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without REMOTE_STORAGE_TOKEN the canonical fallback must apply."""
    monkeypatch.setattr(
        settings, "REMOTE_STORAGE_URL", "http://backend.internal:8000/api/internal"
    )
    monkeypatch.setattr(settings, "REMOTE_STORAGE_TOKEN", "")
    monkeypatch.setattr(settings, "INTERNAL_SERVICE_TOKEN", "internal-token")
    httpx_mock.add_response(
        url="http://backend.internal:8000/api/internal/tables/query",
        json=BACKEND_RESPONSE,
    )

    result = await _query(_make_tool())

    request = httpx_mock.get_request()
    assert result["total_count"] == 1
    assert request.headers["Authorization"] == "Bearer internal-token"


@pytest.mark.asyncio
async def test_query_omits_authorization_when_no_token_configured(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No token configured must fail closed: no Authorization header is sent."""
    monkeypatch.setattr(
        settings, "REMOTE_STORAGE_URL", "http://backend.internal:8000/api/internal"
    )
    monkeypatch.setattr(settings, "REMOTE_STORAGE_TOKEN", "")
    monkeypatch.setattr(settings, "INTERNAL_SERVICE_TOKEN", "")
    httpx_mock.add_response(
        url="http://backend.internal:8000/api/internal/tables/query",
        json=BACKEND_RESPONSE,
    )

    await _query(_make_tool())

    request = httpx_mock.get_request()
    assert "Authorization" not in request.headers


@pytest.mark.asyncio
async def test_query_sends_token_only_to_backend_configured_authority(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recorded reality: the outbound URL authority is governed entirely by
    REMOTE_STORAGE_URL (with the /api/internal suffix stripped) or
    BACKEND_API_URL. This test pins the authority so a future change in token
    routing cannot silently shift where the credential is sent. Host
    allow-listing / egress policy is BUG-005 and out of scope here.
    """
    monkeypatch.setattr(
        settings, "REMOTE_STORAGE_URL", "https://tables.internal.example.com/api/internal"
    )
    monkeypatch.setattr(settings, "REMOTE_STORAGE_TOKEN", "remote-token")
    monkeypatch.setattr(settings, "INTERNAL_SERVICE_TOKEN", "internal-token")
    httpx_mock.add_response(
        url="https://tables.internal.example.com/api/internal/tables/query",
        json=BACKEND_RESPONSE,
    )

    await _query(_make_tool())

    request = httpx_mock.get_request()
    assert request.url == "https://tables.internal.example.com/api/internal/tables/query"
    assert request.headers["Authorization"] == "Bearer remote-token"
