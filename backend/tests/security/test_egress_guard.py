# SPDX-FileCopyrightText: 2026 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Egress invariant: user-controlled outbound URLs must pass the canonical
outbound security policy before any network side effect, and upstream
response bodies must never be relayed verbatim to the caller.

The network-dependent tests use a real local HTTP listener so the guard is
exercised through the production httpx stack; "private" here is loopback,
standing in for intranet / metadata endpoints.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.core.config import settings
from app.services import egress_guard
from app.services.egress_guard import (
    guarded_httpx_client,
    validate_outbound_url,
    validate_redirect_hop,
)

INTERNAL_BODY_MARKER = "SECRET-INTERNAL-SERVICE-BODY"


class _Listener(BaseHTTPRequestHandler):
    hits: list = []
    mode: str = "json-500"

    def _respond(self, status: int, body: bytes, headers: dict | None = None):
        self.send_response(status)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        _Listener.hits.append(self.path)
        if _Listener.mode == "redirect-private":
            body = b""
            self._respond(302, body, {"Location": "http://10.255.0.1/redirected"})
            return
        if _Listener.mode == "json-200":
            body = json.dumps(
                {
                    "name": "intranet-app",
                    "mode": "chat",
                    "secret_field": INTERNAL_BODY_MARKER,
                    "debug_dump": INTERNAL_BODY_MARKER,
                }
            ).encode()
            self._respond(200, body, {"Content-Type": "application/json"})
            return
        body = json.dumps({"marker": INTERNAL_BODY_MARKER}).encode()
        self._respond(500, body, {"Content-Type": "application/json"})


@pytest.fixture()
def listener():
    _Listener.hits = []
    _Listener.mode = "json-500"
    server = HTTPServer(("127.0.0.1", 0), _Listener)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


@pytest.fixture
def allow_loopback(monkeypatch: pytest.MonkeyPatch):
    def _allow(*entries: str) -> None:
        monkeypatch.setattr(
            settings, "EGRESS_PRIVATE_NETWORK_ALLOWLIST", ",".join(entries)
        )

    return _allow


# ---------------------------------------------------------------------------
# Unit contract of the guard itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/x",
        "http://10.0.0.3/x",
        "http://172.16.1.1/x",
        "http://192.168.1.4/x",
        "http://169.254.169.254/latest/meta-data/",
        "http://0.0.0.0/x",
        "http://[::1]/x",
        "http://[::ffff:127.0.0.1]/x",
        "http://user:pass@10.0.0.1/x",
        "http://token@192.168.0.9/x",
        "ftp://example.com/file",
        "file:///etc/passwd",
    ],
)
def test_blocked_urls(url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("DNS resolution must not be needed for blocked hosts")

    monkeypatch.setattr(egress_guard.socket, "getaddrinfo", _fail)
    with pytest.raises(HTTPException) as excinfo:
        validate_outbound_url(url)
    assert excinfo.value.status_code == 400


def test_localhost_resolving_to_loopback_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'localhost' resolves via the hosts file; simulate that resolution."""
    import socket as socket_module

    def _resolve(host, port, **kwargs):  # noqa: ANN002, ANN003
        return [
            (
                socket_module.AF_INET,
                socket_module.SOCK_STREAM,
                6,
                "",
                ("127.0.0.1", 0),
            )
        ]

    monkeypatch.setattr(egress_guard.socket, "getaddrinfo", _resolve)
    with pytest.raises(HTTPException):
        validate_outbound_url("http://localhost/x")


def test_dns_failure_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def _gaierror(*args, **kwargs):  # noqa: ANN002, ANN003
        raise OSError("resolution failed")

    monkeypatch.setattr(egress_guard.socket, "getaddrinfo", _gaierror)
    with pytest.raises(HTTPException):
        validate_outbound_url("http://does-not-resend.invalid/x")


def test_public_address_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket as socket_module

    def _resolve(host, port, **kwargs):  # noqa: ANN002, ANN003
        return [
            (
                socket_module.AF_INET,
                socket_module.SOCK_STREAM,
                6,
                "",
                ("93.184.216.34", 0),
            )
        ]

    monkeypatch.setattr(egress_guard.socket, "getaddrinfo", _resolve)
    validate_outbound_url("https://example.com/owner/repo")


def test_multi_address_with_unsafe_result_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import socket as socket_module

    def _resolve(host, port, **kwargs):  # noqa: ANN002, ANN003
        return [
            (
                socket_module.AF_INET,
                socket_module.SOCK_STREAM,
                6,
                "",
                ("93.184.216.34", 0),
            ),
            (
                socket_module.AF_INET,
                socket_module.SOCK_STREAM,
                6,
                "",
                ("10.0.0.7", 0),
            ),
        ]

    monkeypatch.setattr(egress_guard.socket, "getaddrinfo", _resolve)
    with pytest.raises(HTTPException):
        validate_outbound_url("https://double-record.example.com/x")


def test_allowlist_hostname_and_cidr(
    monkeypatch: pytest.MonkeyPatch,
    allow_loopback,
) -> None:
    allow_loopback("internal-gitea.example.com", "10.0.0.0/8")

    def _deny(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("allowlisted hostname must skip resolution")

    monkeypatch.setattr(egress_guard.socket, "getaddrinfo", _deny)

    validate_outbound_url("http://internal-gitea.example.com:3000/x")

    import socket as socket_module

    def _resolve(host, port, **kwargs):  # noqa: ANN002, ANN003
        assert host == "10.1.2.3"
        return [
            (
                socket_module.AF_INET,
                socket_module.SOCK_STREAM,
                6,
                "",
                ("10.1.2.3", 0),
            )
        ]

    monkeypatch.setattr(egress_guard.socket, "getaddrinfo", _resolve)
    validate_outbound_url("http://10.1.2.3/api")


def test_redirect_hop_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket as socket_module

    def _resolve(host, port, **kwargs):  # noqa: ANN002, ANN003
        assert host == "93.184.216.34"
        return [
            (
                socket_module.AF_INET,
                socket_module.SOCK_STREAM,
                6,
                "",
                ("93.184.216.34", 0),
            )
        ]

    monkeypatch.setattr(egress_guard.socket, "getaddrinfo", _resolve)
    with pytest.raises(HTTPException):
        validate_redirect_hop("https://93.184.216.34/start", "http://127.0.0.1/x")
    validate_redirect_hop("https://93.184.216.34/start", "/relative/path")


def test_guarded_client_refuses_private_redirect(listener, allow_loopback) -> None:
    allow_loopback("127.0.0.1")
    _Listener.mode = "redirect-private"

    with guarded_httpx_client(timeout=5.0) as client:
        with pytest.raises(HTTPException):
            client.get(f"http://127.0.0.1:{listener}/start")

    # Only the initial (allowlisted) request was made; the private redirect
    # target was never contacted.
    assert _Listener.hits == ["/start"]


# ---------------------------------------------------------------------------
# Integration: git skill scan (BUG-005)
# ---------------------------------------------------------------------------


def test_git_scan_blocked_before_network_side_effect(listener) -> None:
    from app.services.git_skill.service import git_skill_service

    with pytest.raises(HTTPException) as excinfo:
        git_skill_service.scan_repository(
            f"http://127.0.0.1:{listener}/acme/skills",
            user_id=1,
            db=None,
        )

    assert "outbound request policy" in excinfo.value.detail
    # The exploit's defining fact — a server-side request to the attacker's
    # host — must no longer happen.
    assert _Listener.hits == []


def test_git_scan_upstream_body_not_reflected(listener, allow_loopback) -> None:
    allow_loopback("127.0.0.1")
    from app.services.git_skill.service import git_skill_service

    with pytest.raises(HTTPException) as excinfo:
        git_skill_service.scan_repository(
            f"http://127.0.0.1:{listener}/acme/skills",
            user_id=1,
            db=None,
        )

    assert INTERNAL_BODY_MARKER not in str(excinfo.value.detail)


# ---------------------------------------------------------------------------
# Integration: Dify app info / parameters (VA-5a)
# ---------------------------------------------------------------------------


@pytest.fixture
def dify_client(allow_loopback):
    from types import SimpleNamespace

    from fastapi import FastAPI

    from app.api.dependencies import get_db
    from app.api.endpoints.adapter import dify
    from app.core import security

    allow_loopback("127.0.0.1")
    app = FastAPI()
    app.include_router(dify.router)
    app.dependency_overrides[security.get_current_user] = lambda: SimpleNamespace(
        id=1, user_name="tester"
    )

    def override_get_db():
        yield None

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client


def test_dify_info_whitelists_upstream_response(
    dify_client: TestClient, listener
) -> None:
    _Listener.mode = "json-200"

    response = dify_client.post(
        "/app/info",
        json={"api_key": "app-testkey", "base_url": f"http://127.0.0.1:{listener}"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body.get("name") == "intranet-app"
    assert body.get("mode") == "chat"
    # Full upstream readback is gone: only whitelisted fields come back.
    assert "secret_field" not in body
    assert "debug_dump" not in body
    assert INTERNAL_BODY_MARKER not in response.text


def test_dify_error_body_not_reflected(dify_client: TestClient, listener) -> None:
    _Listener.mode = "json-500"

    response = dify_client.post(
        "/app/info",
        json={"api_key": "app-testkey", "base_url": f"http://127.0.0.1:{listener}"},
    )

    assert response.status_code == 502
    assert INTERNAL_BODY_MARKER not in response.text
    assert "marker" not in response.text


def test_dify_private_base_url_blocked_without_allowlist(
    listener,
) -> None:
    from types import SimpleNamespace

    from fastapi import FastAPI

    from app.api.dependencies import get_db
    from app.api.endpoints.adapter import dify
    from app.core import security

    app = FastAPI()
    app.include_router(dify.router)
    app.dependency_overrides[security.get_current_user] = lambda: SimpleNamespace(
        id=1, user_name="tester"
    )

    def override_get_db():
        yield None

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        response = client.post(
            "/app/info",
            json={
                "api_key": "app-testkey",
                "base_url": f"http://127.0.0.1:{listener}",
            },
        )

    assert response.status_code == 400
    assert "outbound request policy" in response.json()["detail"]
    assert _Listener.hits == []
