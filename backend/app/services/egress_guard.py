# SPDX-FileCopyrightText: 2026 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Canonical outbound (egress) security policy for user-controlled URLs.

Wegent is self-hostable and explicitly supports private infrastructure
(self-hosted Gitea/GitLab, self-hosted Dify, local model services), so the
policy is NOT "private IP == always illegal". Instead:

- Default deny: user-controlled outbound requests must not reach private,
  loopback, link-local, reserved, or unspecified networks.
- Administrator escape hatch: ``EGRESS_PRIVATE_NETWORK_ALLOWLIST`` (env)
  lists exact hostnames or CIDRs that are trusted for this deployment.
  The allowlist is deployment configuration and can never be influenced
  by request parameters.

Egress invariants enforced here:

- http/https schemes only
- hostname resolves (fail closed on DNS failure)
- every resolved address (IPv4 and IPv6, including IPv4-mapped IPv6) is
  checked against the blocked networks and the administrator allowlist
- redirect hops are revalidated (use :func:`guarded_httpx_client`)

Known limitation (documented in the security PR): this module validates
hostnames before the HTTP client connects, so a DNS-rebinding / TOCTOU
window remains between validation and the client's own resolution. Full
connect-time binding requires a custom transport and is out of scope.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
from fastapi import HTTPException

from app.core.config import settings

logger = logging.getLogger(__name__)

_ALLOWED_SCHEMES = ("http", "https")


@dataclass(frozen=True)
class _Allowlist:
    hostnames: frozenset[str]
    networks: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]


def _parse_allowlist() -> _Allowlist:
    raw = (settings.EGRESS_PRIVATE_NETWORK_ALLOWLIST or "").strip()
    hostnames: set[str] = set()
    networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    if not raw:
        return _Allowlist(frozenset(hostnames), tuple(networks))

    for entry in raw.split(","):
        entry = entry.strip().lower()
        if not entry:
            continue
        try:
            networks.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            hostnames.add(entry)
    return _Allowlist(frozenset(hostnames), tuple(networks))


def _iter_resolved_addresses(hostname: str):
    """Yield every address the hostname refers to (IPv4 and IPv6).

    IP-literal hostnames are parsed directly; real hostnames go through the
    resolver, yielding every A/AAAA result.
    """
    try:
        yield ipaddress.ip_address(hostname)
        return
    except ValueError:
        pass

    results = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    for family, _type, _proto, _canonname, sockaddr in results:
        yield ipaddress.ip_address(sockaddr[0])


def _is_blocked_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_unspecified
        or address.is_multicast
    )


def _address_in_allowlist(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
    allowlist: _Allowlist,
) -> bool:
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return any(address in network for network in allowlist.networks)


def _raise_blocked(target: str, reason: str) -> None:
    logger.warning("Outbound request blocked: %s (%s)", target, reason)
    raise HTTPException(
        status_code=400,
        detail="Target host is not allowed by the outbound request policy",
    )


def validate_outbound_url(url: str) -> str:
    """Validate a user-controlled outbound URL against the egress policy.

    Returns the validated URL unchanged when allowed; raises HTTPException
    when the URL must not be reached.
    """
    parsed = urlparse(url)

    if parsed.scheme not in _ALLOWED_SCHEMES:
        _raise_blocked(url, f"scheme {parsed.scheme!r} not allowed")

    hostname = parsed.hostname
    if not hostname:
        _raise_blocked(url, "missing hostname")

    allowlist = _parse_allowlist()

    if hostname in allowlist.hostnames:
        return url

    try:
        addresses = list(_iter_resolved_addresses(hostname))
    except (socket.gaierror, OSError, ValueError):
        _raise_blocked(url, "hostname resolution failed")

    if not addresses:
        _raise_blocked(url, "hostname resolved to no addresses")

    for address in addresses:
        if _address_in_allowlist(address, allowlist):
            continue
        if _is_blocked_address(address):
            _raise_blocked(url, f"resolved address {address} is in a blocked network")

    return url


def validate_redirect_hop(current_url: str, location: str) -> str:
    """Resolve a redirect ``Location`` against the current URL and validate it.

    Returns the absolute redirect target when allowed; raises HTTPException
    otherwise.
    """
    target = urljoin(current_url, location)
    return validate_outbound_url(target)


def _check_redirect_hop(response: httpx.Response) -> None:
    """httpx response hook that revalidates every redirect target.

    Kept synchronous: guarded_httpx_client creates a sync Client, whose
    event hooks must be plain functions.
    """
    if 300 <= response.status_code < 400:
        location = response.headers.get("location")
        if location:
            validate_redirect_hop(str(response.request.url), location)


def guarded_httpx_client(**kwargs) -> httpx.Client:
    """Create an httpx.Client that enforces the egress policy on every hop.

    Use this for any request whose URL is (partially) user-controlled. Every
    redirect hop is revalidated before the client follows it. The caller's
    event hook mapping is copied, never mutated.
    """
    kwargs.setdefault("follow_redirects", True)
    caller_hooks = kwargs.pop("event_hooks", None) or {}
    hooks = {name: list(hook_list) for name, hook_list in caller_hooks.items()}
    hooks.setdefault("response", []).append(_check_redirect_hop)
    kwargs["event_hooks"] = hooks
    return httpx.Client(**kwargs)
