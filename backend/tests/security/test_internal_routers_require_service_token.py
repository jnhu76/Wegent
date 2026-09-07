# SPDX-FileCopyrightText: 2026 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Internal boundary invariant: sensitive /internal endpoints require the
internal service token. Anonymous or invalid callers must be rejected before
any handler logic runs (no DB mutation, no secret readback)."""

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings

INTERNAL_TOKEN = "test-internal-token"


@pytest.fixture(autouse=True)
def configure_internal_service_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "INTERNAL_SERVICE_TOKEN", INTERNAL_TOKEN)


# (method, path, json body) covering every router secured by this campaign.
# Bodies are minimal on purpose: a 422 from body validation with a valid token
# proves the request passed the auth dependency, while the same request without
# a token must be rejected with 401 before reaching the handler.
PROTECTED_ROUTES = [
    ("GET", "/api/internal/services/health", None),
    ("POST", "/api/internal/services/update", {}),
    ("GET", "/api/internal/services/000", None),
    ("POST", "/api/internal/callback", {}),
    ("POST", "/api/internal/callback/batch", {}),
    ("GET", "/api/internal/bots/any-bot/mcp", None),
    ("POST", "/api/internal/tables/query", {}),
    ("POST", "/api/internal/workspace-archives/1/archive-sandbox", {}),
    ("POST", "/api/internal/workspace-archives/1/restore-sandbox", {}),
    ("GET", "/api/internal/workspace-archives/1/download-url", None),
]


@pytest.mark.parametrize("method,path,body", PROTECTED_ROUTES)
def test_internal_routes_reject_anonymous_callers(
    test_client: TestClient, method: str, path: str, body: dict | None
) -> None:
    response = test_client.request(method, path, json=body)

    assert response.status_code == 401


@pytest.mark.parametrize("method,path,body", PROTECTED_ROUTES)
def test_internal_routes_reject_invalid_tokens(
    test_client: TestClient, method: str, path: str, body: dict | None
) -> None:
    response = test_client.request(
        method,
        path,
        json=body,
        headers={"Authorization": "Bearer not-the-real-token"},
    )

    assert response.status_code == 401


@pytest.mark.parametrize("method,path,body", PROTECTED_ROUTES)
def test_internal_routes_accept_valid_internal_token(
    test_client: TestClient, method: str, path: str, body: dict | None
) -> None:
    response = test_client.request(
        method,
        path,
        json=body,
        headers={"Authorization": f"Bearer {INTERNAL_TOKEN}"},
    )

    assert response.status_code != 401


def test_bot_mcp_readback_returns_empty_for_unknown_bot(
    test_client: TestClient,
) -> None:
    """Positive read path: authenticated caller gets a well-formed answer and
    an unknown bot never leaks secret material."""
    response = test_client.get(
        "/api/internal/bots/does-not-exist/mcp?namespace=default",
        headers={"Authorization": f"Bearer {INTERNAL_TOKEN}"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "bot_name": "does-not-exist",
        "bot_namespace": "default",
        "mcp_servers": {},
    }


def test_services_health_succeeds_with_internal_token(
    test_client: TestClient,
) -> None:
    response = test_client.get(
        "/api/internal/services/health",
        headers={"Authorization": f"Bearer {INTERNAL_TOKEN}"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
