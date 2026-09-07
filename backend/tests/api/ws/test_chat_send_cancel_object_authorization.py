# SPDX-FileCopyrightText: 2026 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Cross-tenant invariant for the chat websocket namespace.

User A supplying User B's task/team/subtask identifiers must not gain access
to B's data nor mutate B's state:

- chat:send follow-up on a task the caller cannot access is rejected before
  any task creation, room join, or dispatch happens.
- chat:send on a new conversation only accepts teams the caller may use
  (same semantics as the REST team detail path).
- chat:cancel on a subtask the caller cannot access is rejected before any
  state mutation or runtime dispatch.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

import app.stores.tasks as task_stores
from app.api.ws import chat_namespace
from app.api.ws.chat_namespace import ChatNamespace, _get_subtask_for_cancel

ATTACKER_ID = 2838
VICTIM_TASK_ID = 390051750105148
VICTIM_TEAM_ID = 267213


def _db_session_context(db):
    @contextmanager
    def _manager():
        yield db

    return _manager()


def _user() -> SimpleNamespace:
    return SimpleNamespace(id=ATTACKER_ID, user_name="attacker")


def _make_namespace() -> ChatNamespace:
    namespace = ChatNamespace()
    namespace.get_session = AsyncMock(
        return_value={
            "user_id": ATTACKER_ID,
            "user_name": "attacker",
            "auth_token": "",
        }
    )
    namespace._check_token_expiry = AsyncMock(return_value=False)
    namespace.enter_room = AsyncMock()
    namespace.emit = AsyncMock()
    return namespace


def _mock_db() -> MagicMock:
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = _user()
    return db


class _AuthzGateProbe(RuntimeError):
    """Raised to prove execution passed the authorization gate."""


@pytest.fixture
def probe_beyond_gate(monkeypatch: pytest.MonkeyPatch):
    """Let authorized requests run until just past the authorization gate."""

    def _install() -> Mock:
        probe = Mock(side_effect=_AuthzGateProbe("passed the gate"))
        monkeypatch.setattr(chat_namespace, "_apply_artifact_node_scope", probe)
        return probe

    return _install


@pytest.fixture
def deny_task_member(monkeypatch: pytest.MonkeyPatch) -> Mock:
    is_member = Mock(return_value=False)
    monkeypatch.setattr(task_stores.task_access_store, "is_member", is_member)
    return is_member


@pytest.fixture
def allow_task_member(monkeypatch: pytest.MonkeyPatch) -> Mock:
    is_member = Mock(return_value=True)
    monkeypatch.setattr(task_stores.task_access_store, "is_member", is_member)
    return is_member


@pytest.fixture
def stub_team_crd(monkeypatch: pytest.MonkeyPatch) -> None:
    team_crd = SimpleNamespace(spec=SimpleNamespace(collaborationModel="standard"))
    monkeypatch.setattr(
        chat_namespace, "Team", SimpleNamespace(model_validate=Mock(return_value=team_crd))
    )


@pytest.mark.asyncio
async def test_followup_denied_for_non_member_before_task_creation(
    monkeypatch: pytest.MonkeyPatch,
    deny_task_member: Mock,
) -> None:
    """Attacker + victim task_id → denied, no create_chat_task, no room join."""
    namespace = _make_namespace()
    victim_task = SimpleNamespace(
        id=VICTIM_TASK_ID, user_id=999, json={"spec": {"teamRef": None}}
    )
    monkeypatch.setattr(
        chat_namespace,
        "_resolve_existing_task_team",
        Mock(
            return_value=(
                victim_task,
                SimpleNamespace(id=VICTIM_TEAM_ID, name="victim-team", json={"spec": {}}),
                None,
            )
        ),
    )
    monkeypatch.setattr(
        "app.services.chat.storage.create_chat_task", Mock()
    )

    db = _mock_db()
    with patch("app.api.ws.chat_namespace.SessionLocal", return_value=db):
        result = await namespace.on_chat_send(
            "sid-attacker",
            {"task_id": VICTIM_TASK_ID, "team_id": VICTIM_TEAM_ID, "message": "hi"},
        )

    assert result == {"error": "Task not found"}
    deny_task_member.assert_called_once_with(
        db, task_id=VICTIM_TASK_ID, user_id=ATTACKER_ID
    )
    namespace.enter_room.assert_not_called()


@pytest.mark.asyncio
async def test_followup_allowed_for_authorized_member(
    monkeypatch: pytest.MonkeyPatch,
    allow_task_member: Mock,
    stub_team_crd: None,
    probe_beyond_gate,
) -> None:
    probe = probe_beyond_gate()

    namespace = _make_namespace()
    task = SimpleNamespace(id=VICTIM_TASK_ID, user_id=ATTACKER_ID, json={"spec": {}})
    monkeypatch.setattr(
        chat_namespace,
        "_resolve_existing_task_team",
        Mock(
            return_value=(
                task,
                SimpleNamespace(id=VICTIM_TEAM_ID, name="bound-team", json={"spec": {}}),
                None,
            )
        ),
    )
    monkeypatch.setattr(
        "app.services.chat.config.is_deep_research_protocol", Mock(return_value=False)
    )
    monkeypatch.setattr(
        "app.services.chat.interactive_forms.validate_interactive_form_answer",
        Mock(return_value=SimpleNamespace(ok=True, error=None, message=None)),
    )

    with patch("app.api.ws.chat_namespace.SessionLocal", return_value=_mock_db()):
        result = await namespace.on_chat_send(
            "sid-owner",
            {"task_id": VICTIM_TASK_ID, "team_id": VICTIM_TEAM_ID, "message": "hi"},
        )

    assert "passed the gate" in result.get("error", "")
    allow_task_member.assert_called_once()
    probe.assert_called_once()


@pytest.mark.asyncio
async def test_new_conversation_denied_for_inaccessible_team(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attacker + victim team_id → denied before task creation."""
    namespace = _make_namespace()
    victim_team = SimpleNamespace(id=VICTIM_TEAM_ID, name="victim-team", json={"spec": {}})
    monkeypatch.setattr(
        chat_namespace,
        "_get_active_team_by_id",
        Mock(return_value=victim_team),
    )
    from app.services.share.team_share_service import team_share_service

    monkeypatch.setattr(
        team_share_service, "get_resource", Mock(return_value=None)
    )
    create_chat_task = Mock()
    monkeypatch.setattr("app.services.chat.storage.create_chat_task", create_chat_task)

    with patch("app.api.ws.chat_namespace.SessionLocal", return_value=_mock_db()):
        result = await namespace.on_chat_send(
            "sid-attacker",
            {"team_id": VICTIM_TEAM_ID, "message": "hi"},
        )

    assert result == {"error": "Team not found"}
    create_chat_task.assert_not_called()
    namespace.enter_room.assert_not_called()


@pytest.mark.asyncio
async def test_new_conversation_allowed_for_shared_team(
    monkeypatch: pytest.MonkeyPatch, stub_team_crd: None, probe_beyond_gate
) -> None:
    """A legitimately shared team member must still be able to chat."""
    probe = probe_beyond_gate()

    namespace = _make_namespace()
    shared_team = SimpleNamespace(
        id=VICTIM_TEAM_ID, name="shared-team", json={"spec": {}}
    )
    monkeypatch.setattr(
        chat_namespace,
        "_get_active_team_by_id",
        Mock(return_value=shared_team),
    )
    from app.services.share.team_share_service import team_share_service

    monkeypatch.setattr(
        team_share_service, "get_resource", Mock(return_value=shared_team)
    )

    with patch("app.api.ws.chat_namespace.SessionLocal", return_value=_mock_db()):
        result = await namespace.on_chat_send(
            "sid-member",
            {"team_id": VICTIM_TEAM_ID, "message": "hi"},
        )

    assert "passed the gate" in result.get("error", "")
    probe.assert_called_once()


def test_get_subtask_for_cancel_denies_non_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subtask = SimpleNamespace(
        id=5, task_id=VICTIM_TASK_ID, status="RUNNING", executor_name="exec-1"
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None
    with (
        patch("app.api.ws.chat_namespace.get_db_session", return_value=_db_session_context(db)),
        patch.object(
            task_stores.subtask_store, "get_by_id", Mock(return_value=subtask)
        ),
        patch.object(
            task_stores.task_access_store, "is_member", Mock(return_value=False)
        ) as is_member,
    ):
        info = _get_subtask_for_cancel(5, ATTACKER_ID)

    assert info is None
    is_member.assert_called_once_with(db, task_id=VICTIM_TASK_ID, user_id=ATTACKER_ID)


def test_get_subtask_for_cancel_returns_info_for_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subtask = SimpleNamespace(
        id=5, task_id=VICTIM_TASK_ID, status="RUNNING", executor_name="exec-1"
    )
    db = MagicMock()
    with (
        patch("app.api.ws.chat_namespace.get_db_session", return_value=_db_session_context(db)),
        patch.object(
            task_stores.subtask_store, "get_by_id", Mock(return_value=subtask)
        ),
        patch.object(
            task_stores.task_access_store, "is_member", Mock(return_value=True)
        ),
    ):
        info = _get_subtask_for_cancel(5, ATTACKER_ID)

    assert info is not None
    assert info["task_id"] == VICTIM_TASK_ID


@pytest.mark.asyncio
async def test_cancel_denied_before_state_mutation_and_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attacker + victim subtask_id → denied, no mutation, no dispatch."""
    namespace = _make_namespace()
    monkeypatch.setattr(
        chat_namespace,
        "_get_subtask_for_cancel",
        Mock(return_value=None),
    )
    mark_cancelling = Mock()
    monkeypatch.setattr(chat_namespace, "_mark_task_and_board_cancelling", mark_cancelling)
    dispatcher = MagicMock()
    dispatcher.cancel = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "app.services.execution.dispatcher.execution_dispatcher", dispatcher
    )

    result = await namespace.on_chat_cancel("sid-attacker", {"subtask_id": 5})

    assert result == {"error": "Subtask not found"}
    mark_cancelling.assert_not_called()
    dispatcher.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_still_works_for_authorized_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _make_namespace()
    monkeypatch.setattr(
        chat_namespace,
        "_get_subtask_for_cancel",
        Mock(
            return_value={
                "id": 5,
                "task_id": VICTIM_TASK_ID,
                "status": "RUNNING",
                "executor_name": None,
            }
        ),
    )
    mark_cancelling = Mock()
    monkeypatch.setattr(chat_namespace, "_mark_task_and_board_cancelling", mark_cancelling)
    dispatcher = MagicMock()
    dispatcher.cancel = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "app.services.execution.dispatcher.execution_dispatcher", dispatcher
    )

    result = await namespace.on_chat_cancel("sid-owner", {"subtask_id": 5})

    assert result == {"success": True}
    mark_cancelling.assert_called_once_with(5)
    dispatcher.cancel.assert_awaited_once()
