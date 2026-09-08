# SPDX-FileCopyrightText: 2026 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.core.config import settings
from app.services.auth.internal_service_token import (
    require_internal_service_token_configured,
    verify_internal_service_token,
)


def _credentials(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def test_missing_authorization_is_rejected_when_token_is_configured(monkeypatch):
    monkeypatch.setattr(settings, "INTERNAL_SERVICE_TOKEN", "test-internal-token")

    with pytest.raises(HTTPException) as exc_info:
        verify_internal_service_token(credentials=None)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Missing authentication token"


def test_invalid_authorization_is_rejected_when_token_is_configured(monkeypatch):
    monkeypatch.setattr(settings, "INTERNAL_SERVICE_TOKEN", "test-internal-token")

    with pytest.raises(HTTPException) as exc_info:
        verify_internal_service_token(credentials=_credentials("invalid-token"))

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid authentication token"


def test_non_ascii_authorization_is_rejected_with_401_not_500(monkeypatch):
    """ASGI servers decode header values as latin-1, so a crafted bearer can
    carry non-ASCII characters. The dependency must answer 401; comparing
    non-ASCII strings with hmac.compare_digest raises TypeError, so the
    comparison must happen on bytes to keep the 401 path."""
    monkeypatch.setattr(settings, "INTERNAL_SERVICE_TOKEN", "test-internal-token")

    with pytest.raises(HTTPException) as exc_info:
        verify_internal_service_token(credentials=_credentials("tökën-ïnvälid"))

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid authentication token"


def test_valid_authorization_is_accepted_when_token_is_configured(monkeypatch):
    monkeypatch.setattr(settings, "INTERNAL_SERVICE_TOKEN", "test-internal-token")

    verify_internal_service_token(credentials=_credentials("test-internal-token"))


def test_unconfigured_internal_service_token_rejects_missing_authorization(monkeypatch):
    monkeypatch.setattr(settings, "INTERNAL_SERVICE_TOKEN", "")

    with pytest.raises(HTTPException) as exc_info:
        verify_internal_service_token(credentials=None)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Internal service token is not configured"


def test_unconfigured_internal_service_token_rejects_present_authorization(monkeypatch):
    monkeypatch.setattr(settings, "INTERNAL_SERVICE_TOKEN", "")

    with pytest.raises(HTTPException) as exc_info:
        verify_internal_service_token(credentials=_credentials("any-token"))

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Internal service token is not configured"


def test_whitespace_internal_service_token_is_unconfigured(monkeypatch):
    monkeypatch.setattr(settings, "INTERNAL_SERVICE_TOKEN", "   ")

    with pytest.raises(HTTPException) as exc_info:
        verify_internal_service_token(credentials=_credentials("any-token"))

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Internal service token is not configured"


def test_startup_config_check_rejects_unconfigured_token(monkeypatch):
    monkeypatch.setattr(settings, "INTERNAL_SERVICE_TOKEN", "")

    with pytest.raises(RuntimeError) as exc_info:
        require_internal_service_token_configured()

    assert "INTERNAL_SERVICE_TOKEN is required" in str(exc_info.value)


def test_startup_config_check_rejects_whitespace_token(monkeypatch):
    monkeypatch.setattr(settings, "INTERNAL_SERVICE_TOKEN", "   ")

    with pytest.raises(RuntimeError) as exc_info:
        require_internal_service_token_configured()

    assert "INTERNAL_SERVICE_TOKEN is required" in str(exc_info.value)


def test_startup_config_check_accepts_configured_token(monkeypatch):
    monkeypatch.setattr(settings, "INTERNAL_SERVICE_TOKEN", "test-internal-token")

    require_internal_service_token_configured()
