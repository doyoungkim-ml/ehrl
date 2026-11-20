"""Stateful execute failures must not resend an already-applied action."""

import json
import pickle
import queue

import pytest
import requests

from rllm.environments.appworld import appworld_env as module
from rllm.environments.appworld.appworld_env import AppWorldEnv
from rllm.environments.base.base_env import UncertainExecutionError, is_transient_env_error


def response(status=200, output=None):
    result = requests.Response()
    result.status_code = status
    result.url = "http://unused.test/execute"
    result._content = json.dumps({"output": output}).encode()
    return result


@pytest.mark.parametrize("failure", [500, 502, 503, requests.ReadTimeout, requests.ConnectionError, requests.exceptions.ChunkedEncodingError, requests.exceptions.ContentDecodingError])
def test_execute_applied_before_failed_response_is_not_replayed(monkeypatch, failure):
    endpoints = []
    applied = []
    sleeps = []

    def post(url, json, timeout, **kwargs):
        endpoint = url.rsplit("/", 1)[-1]
        endpoints.append(endpoint)
        if endpoint == "execute":
            applied.append(json["code"])
            if isinstance(failure, int):
                return response(failure)
            raise failure("response failed after action was applied")
        assert endpoint == "close", "Uncertain execution must not query completion or reward"
        return response()

    monkeypatch.setattr(module.requests, "post", post)
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    env = AppWorldEnv(task={"task_id": "synthetic"})
    observation, reward, done, info = env.step("state += 1; complete_task()")

    assert applied == ["state += 1; complete_task()"]
    assert endpoints == ["execute", "close"]
    assert sleeps == []
    assert reward == 0.0 and done and env.done
    assert info["termination"] == "ENV_EXECUTION_UNCERTAIN"
    assert info["execution_uncertain"] is True
    assert info["execution_outcome"] == observation["execution_outcome"] == "unknown"
    assert info["execute_retry_attempted"] is False
    assert info["execution_http_status"] == (failure if isinstance(failure, int) else None)
    assert "may already have run" in observation["error"]
    assert observation["success"] is False
    assert env.execution_in_flight is False
    # A caller cannot accidentally send another action before resetting the world.
    env.step("state += 1")
    assert len(applied) == 1


def test_uncertain_execute_is_not_classified_as_retryable_through_its_cause(monkeypatch):
    def post(*args, **kwargs):
        raise requests.ReadTimeout("response lost")

    monkeypatch.setattr(module.requests, "post", post)
    env = AppWorldEnv(task={"task_id": "synthetic"})
    with pytest.raises(UncertainExecutionError) as caught:
        env._post("/execute", json={"code": "state += 1"})
    assert isinstance(caught.value.__cause__, requests.ReadTimeout)
    assert not env.is_retryable_error(caught.value)
    assert not is_transient_env_error(caught.value)
    assert caught.value.info["execution_outcome"] == "unknown"
    assert pickle.loads(pickle.dumps(caught.value)).info == caught.value.info
    # Reset errors remain retryable independently of stateful execute errors.
    assert env.is_retryable_error(requests.ReadTimeout("initialization timed out"))


@pytest.mark.parametrize("endpoint", ["/initialize", "/task_completed", "/evaluate", "/close"])
@pytest.mark.parametrize("failure", [500, requests.ReadTimeout, requests.ConnectionError])
def test_other_endpoint_retry_behavior_is_preserved(monkeypatch, endpoint, failure):
    calls = []
    sleeps = []

    def post(url, **kwargs):
        calls.append(url)
        if len(calls) < 3:
            if isinstance(failure, int):
                return response(failure)
            raise failure("temporary failure")
        return response(output="ok")

    monkeypatch.setattr(module.requests, "post", post)
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    env = AppWorldEnv(task={"task_id": "synthetic"})
    assert env._post(endpoint).json()["output"] == "ok"
    assert len(calls) == 3 and sleeps == [0.5, 1.0]
    assert env.execution_uncertainty is None


def test_known_client_error_keeps_existing_nonterminal_failure_behavior(monkeypatch):
    calls = []

    def post(url, **kwargs):
        calls.append(url)
        return response(400)

    monkeypatch.setattr(module.requests, "post", post)
    env = AppWorldEnv(task={"task_id": "synthetic"})
    _, reward, done, info = env.step("invalid request")
    assert len(calls) == 1 and reward == 0.0 and not done
    assert info == {"reason": "execution_error"}
    assert env.execution_uncertainty is None


def test_normal_success_keeps_existing_completion_reward(monkeypatch):
    endpoints = []

    def post(url, **kwargs):
        endpoint = url.rsplit("/", 1)[-1]
        endpoints.append(endpoint)
        return response(output={"execute": "normal output", "task_completed": True, "evaluate": {"success": True}, "close": None}[endpoint])

    monkeypatch.setattr(module.requests, "post", post)
    env = AppWorldEnv(task={"task_id": "synthetic"})
    observation, reward, done, info = env.step("complete_task()")
    assert endpoints == ["execute", "task_completed", "evaluate", "close"]
    assert reward == 1.0 and done and observation["output"] == "normal output"
    assert info == {"step": 1} and env.execution_uncertainty is None


def test_uncertain_task_closes_before_returning_its_server_port(monkeypatch):
    pool = queue.Queue()
    pool.put(9000)
    monkeypatch.setattr(module, "_server_port_pool", pool)
    env = AppWorldEnv(task={"task_id": "synthetic"}, n_servers=2)
    env._pool_port = pool.get_nowait()

    def post(url, **kwargs):
        if url.endswith("/execute"):
            return response(500)
        assert url.endswith("/close") and pool.empty()
        return response()

    monkeypatch.setattr(module.requests, "post", post)
    assert env.step("state += 1")[2] is True
    assert pool.get_nowait() == 9000 and env._pool_port is None
    env.close()
    assert pool.empty(), "Closing twice must not return the same port twice"


def test_new_world_reset_clears_previous_uncertainty(monkeypatch):
    env = AppWorldEnv(task={"task_id": "synthetic"})
    env.mark_execution_uncertain("HTTPError", 500)
    monkeypatch.setattr(module.requests, "post", lambda *args, **kwargs: response(output={"instruction": "Synthetic task"}))
    env.reset()
    assert not env.done and env.execution_uncertainty is None and env.current_turn == 0


def test_reset_cannot_reuse_world_with_execute_still_in_flight():
    env = AppWorldEnv(task={"task_id": "synthetic"})
    env._execution_in_flight = True
    with pytest.raises(UncertainExecutionError):
        env.reset()
    assert env.execution_uncertainty["execution_error_type"] == "ExecutionStillInFlight"


@pytest.mark.parametrize("status", [500, 307])
def test_real_http_failure_after_action_is_applied_sends_execute_once(status):
    """A local HTTP route counts applied actions, including redirect replay."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    applied = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if self.path.startswith("/execute"):
                applied.append(self.path)
                self.send_response(status)
                if status == 307:
                    self.send_header("Location", "/execute-again")
                self.end_headers()
                self.wfile.write(b'{"error":"response failed after state changed"}')
            else:
                assert self.path == "/close"
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"output":null}')

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        env = AppWorldEnv(task={"task_id": "synthetic"}, remote_environment_url=f"http://127.0.0.1:{server.server_port}")
        _, reward, done, info = env.step("state += 1")
        assert applied == ["/execute"]
        assert reward == 0 and done and info["execution_uncertain"]
        assert info["execution_http_status"] == status
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("body", [b"not json", b"{}", b"[]", b"null", b'"unexpected"'])
def test_success_status_without_usable_output_is_flagged_uncertain(monkeypatch, body):
    calls = []

    def post(url, **kwargs):
        calls.append(url.rsplit("/", 1)[-1])
        result = response()
        if url.endswith("/execute"):
            result._content = body
        return result

    monkeypatch.setattr(module.requests, "post", post)
    env = AppWorldEnv(task={"task_id": "synthetic"})
    _, reward, done, info = env.step("state += 1")
    assert calls == ["execute", "close"]
    assert reward == 0 and done and info["execution_outcome"] == "unknown"
    env.step("state += 1")
    assert calls.count("execute") == 1
