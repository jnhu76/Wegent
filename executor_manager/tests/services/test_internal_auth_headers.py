# SPDX-FileCopyrightText: 2026 Weibo, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Caller-side contract: every executor_manager -> backend internal API call
must carry the internal service token as a Bearer Authorization header."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from executor_manager.clients import callback_client as callback_client_module
from executor_manager.clients.callback_client import CallbackClient
from executor_manager.config import config as em_config
from executor_manager.executors.docker import executor as docker_executor_module
from executor_manager.executors.docker.executor import DockerExecutor
from executor_manager.routers import routers
from executor_manager.services.sandbox import manager as sandbox_manager_module


@pytest.fixture
def configure_token(monkeypatch):
    def _configure(token: str) -> None:
        monkeypatch.setattr(em_config, "INTERNAL_SERVICE_TOKEN", token)

    return _configure


def _capture_client(mocker, captured):
    client = MagicMock()
    response = MagicMock()
    response.status_code = 200
    client.post = mocker.AsyncMock(return_value=response)

    def _capture_post(url, **kwargs):
        captured.append({"url": url, **kwargs})
        return response

    client.post = mocker.AsyncMock(side_effect=_capture_post)

    cm = MagicMock()
    cm.__aenter__ = mocker.AsyncMock(return_value=client)
    cm.__aexit__ = mocker.AsyncMock(return_value=None)
    return cm


@pytest.mark.asyncio
async def test_callback_client_sends_internal_token(mocker, configure_token):
    configure_token("em-secret-token")
    captured = []
    mocker.patch.object(
        callback_client_module,
        "traced_async_client",
        return_value=_capture_client(mocker, captured),
    )

    result = await CallbackClient().send_error(
        task_id=1, subtask_id=2, error_message="boom"
    )

    assert result is True
    assert captured[0]["headers"] == {"Authorization": "Bearer em-secret-token"}


def test_internal_service_auth_headers_empty_without_token(configure_token):
    configure_token("")
    assert em_config.internal_service_auth_headers() == {}


@pytest.mark.asyncio
async def test_validation_status_bridge_sends_internal_token(
    mocker, configure_token
):
    configure_token("em-secret-token")
    captured = []
    mocker.patch.object(
        routers,
        "traced_async_client",
        return_value=_capture_client(mocker, captured),
    )

    await routers._update_validation_status_from_callback(
        validation_id="val-1",
        event_type="response.completed",
        event_data={
            "executor_name": "wegent-executor-1",
            "data": {"response": {"output": []}},
        },
    )

    assert captured[0]["url"].endswith("/api/shells/validation-status/val-1")
    assert captured[0]["headers"] == {"Authorization": "Bearer em-secret-token"}


@pytest.mark.asyncio
async def test_validation_stage_report_sends_internal_token(mocker, configure_token):
    configure_token("em-secret-token")
    captured = []
    sync_cm = MagicMock()
    sync_client = MagicMock()
    sync_response = MagicMock()
    sync_response.status_code = 200
    sync_client.post.side_effect = lambda url, **kwargs: (
        captured.append({"url": url, **kwargs}),
        sync_response,
    )[1]
    sync_cm.__enter__ = MagicMock(return_value=sync_client)
    sync_cm.__exit__ = MagicMock(return_value=None)
    mocker.patch.object(
        docker_executor_module, "traced_sync_client", return_value=sync_cm
    )

    task = {"metadata": {"task_id": 1, "validation_params": {"validation_id": "val-9"}}}
    DockerExecutor._report_validation_stage(
        MagicMock(),
        task,
        stage="starting_container",
        status="running",
        progress=40,
        message="Container started",
    )

    assert captured[0]["url"].endswith("/api/shells/validation-status/val-9")
    assert captured[0]["headers"] == {"Authorization": "Bearer em-secret-token"}


def _stage_report_executor(mocker, keep_failed: bool = False) -> DockerExecutor:
    executor = DockerExecutor.__new__(DockerExecutor)
    mocker.patch.object(
        executor,
        "_should_keep_failed_validation_container",
        return_value=keep_failed,
    )
    return executor


def _stage_report_client(mocker, captured, status_code: int):
    sync_client = MagicMock()
    sync_response = MagicMock()
    sync_response.status_code = status_code
    sync_response.text = f"status={status_code}"
    sync_client.post.side_effect = lambda url, **kwargs: (
        captured.append({"url": url, **kwargs}),
        sync_response,
    )[1]
    sync_cm = MagicMock()
    sync_cm.__enter__ = MagicMock(return_value=sync_client)
    sync_cm.__exit__ = MagicMock(return_value=None)
    mocker.patch.object(
        docker_executor_module, "traced_sync_client", return_value=sync_cm
    )


_STAGE_TASK = {
    "metadata": {"task_id": 1, "validation_params": {"validation_id": "val-9"}}
}


def test_validation_stage_report_404_cleans_up_local_container(mocker, configure_token):
    configure_token("em-secret-token")
    captured = []
    _stage_report_client(mocker, captured, status_code=404)
    delete_container = mocker.patch.object(docker_executor_module, "delete_container")
    executor = _stage_report_executor(mocker, keep_failed=False)

    executor._report_validation_stage(
        _STAGE_TASK,
        stage="starting_container",
        status="running",
        progress=50,
        message="Container started",
        executor_name="wegent-executor-1",
    )

    delete_container.assert_called_once_with("wegent-executor-1")


def test_validation_stage_report_404_honors_keep_failed_flag(mocker, configure_token):
    configure_token("em-secret-token")
    captured = []
    _stage_report_client(mocker, captured, status_code=404)
    delete_container = mocker.patch.object(docker_executor_module, "delete_container")
    executor = _stage_report_executor(mocker, keep_failed=True)

    executor._report_validation_stage(
        _STAGE_TASK,
        stage="starting_container",
        status="running",
        progress=50,
        message="Container started",
        executor_name="wegent-executor-1",
    )

    delete_container.assert_not_called()


def test_validation_stage_report_404_without_container_name_skips_cleanup(mocker, configure_token):
    configure_token("em-secret-token")
    captured = []
    _stage_report_client(mocker, captured, status_code=404)
    delete_container = mocker.patch.object(docker_executor_module, "delete_container")
    executor = _stage_report_executor(mocker, keep_failed=False)

    executor._report_validation_stage(
        _STAGE_TASK,
        stage="starting_container",
        status="running",
        progress=50,
        message="Container started",
    )

    delete_container.assert_not_called()


def test_validation_stage_report_server_error_does_not_cleanup(mocker, configure_token):
    configure_token("em-secret-token")
    captured = []
    _stage_report_client(mocker, captured, status_code=502)
    delete_container = mocker.patch.object(docker_executor_module, "delete_container")
    executor = _stage_report_executor(mocker, keep_failed=False)

    executor._report_validation_stage(
        _STAGE_TASK,
        stage="starting_container",
        status="running",
        progress=50,
        message="Container started",
        executor_name="wegent-executor-1",
    )

    delete_container.assert_not_called()


def test_create_instance_reports_pulling_image_before_docker_run(mocker, configure_token):
    configure_token("em-secret-token")
    executor = DockerExecutor.__new__(DockerExecutor)
    order = []
    run_mock = MagicMock(
        side_effect=lambda *args, **kwargs: order.append("docker_run")
        or SimpleNamespace(stdout="container-id\n")
    )
    executor.subprocess = MagicMock(run=run_mock)
    mocker.patch.object(executor, "_get_base_image_from_task", return_value=None)
    mocker.patch.object(
        executor, "_get_executor_image", return_value="wegent-executor:latest"
    )
    mocker.patch.object(
        executor, "_prepare_docker_command", return_value=["docker", "run", "img"]
    )
    mocker.patch.object(executor, "register_task_for_heartbeat")
    report_mock = MagicMock(
        side_effect=lambda task, **kwargs: order.append(f"report:{kwargs['stage']}")
    )
    mocker.patch.object(executor, "_report_validation_stage", report_mock)

    task = {"metadata": {"task_id": 1, "subtask_id": 2, "type": "validation"}}
    executor.create_instance(task, {"task_id": 1, "subtask_id": 2}, "exec-1")

    assert order == ["report:pulling_image", "docker_run", "report:starting_container"]
    assert report_mock.call_args_list[0].kwargs["executor_name"] == "exec-1"


@pytest.mark.asyncio
async def test_workspace_archive_callback_sends_internal_token(
    mocker, configure_token
):
    configure_token("em-secret-token")
    captured = []

    class _Sandbox(SimpleNamespace):
        pass

    sandbox = _Sandbox(sandbox_id="sb-1", metadata={})
    manager = sandbox_manager_module.SandboxManager()
    mocker.patch.object(
        sandbox_manager_module.httpx.AsyncClient,
        "__aenter__",
        mocker.AsyncMock(
            return_value=_ArchiveClient(mocker, captured, mocker.MagicMock())
        ),
    )
    mocker.patch.object(
        sandbox_manager_module.httpx.AsyncClient,
        "__aexit__",
        mocker.AsyncMock(return_value=None),
    )

    result = await manager._post_workspace_archive_callback(
        url="http://backend/api/internal/workspace-archives/1/archive-sandbox",
        payload={"executor_name": "exec-1"},
        action="archive",
        sandbox=sandbox,
    )

    assert result is True
    assert captured[0]["headers"] == {
        "Content-Type": "application/json",
        "Authorization": "Bearer em-secret-token",
    }


class _ArchiveClient:
    def __init__(self, mocker, captured, response):
        self._captured = captured
        self._response = response
        self._response.raise_for_status.return_value = None
        self._response.json.return_value = {"success": True}
        self.post = mocker.AsyncMock(side_effect=self._post)

    async def _post(self, url, **kwargs):
        self._captured.append({"url": url, **kwargs})
        return self._response
