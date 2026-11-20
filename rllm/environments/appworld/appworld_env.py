"""
AppWorld Environment — Pure HTTP Client.

Talks to an AppWorld server started separately (e.g., in a conda env with
pydantic v1 and all AppWorld dependencies). This file has zero appworld imports.

No patches. No locks. No monkey-patching. No pydantic conflicts.

Architecture:
    The AgentExecutionEngine runs each env's step() in a ThreadPoolExecutor
    thread via run_in_executor(). This means:
    - step() must be sync (matches BaseEnv interface)
    - The engine provides async concurrency across threads
    - Each env instance is independent (own server_url, own task_id)
    - No shared state → no lock needed → parallel execution is safe

Server:
    Start the AppWorld server separately:
        conda activate appworld && appworld serve environment --port 9000

    Or via Python:
        python -c "from appworld.serve.environment import run; run(port=9000)"

    The server exposes 5 endpoints:
        POST /initialize    — create a new task world
        POST /execute       — run Python code in the world's shell
        POST /task_completed — check if agent called complete_task()
        POST /evaluate      — run tests and get reward
        POST /close         — clean up the task world

Multi-server support:
    AppWorld servers are stateful — each handles one task at a time. When
    running with n_servers > 1, a port pool ensures no two concurrent envs
    share a server.

    Port lifecycle:
        reset()  → pool.get()   (checkout — blocks until a port is free)
        step()   → pool.put()   (when done=True — frees port BEFORE semaphore releases)
        close()  → pool.put()   (backup, if step didn't already return it)
        __del__  → pool.put()   (safety net for leaked envs)

    Why checkout in reset(), not from_dict()?
        In VeRL/GRPO, from_dict() is called for ALL envs at batch init time
        (inside init_envs_and_agents), not during trajectory execution. All
        ports would be checked out at once and never returned. reset() runs
        inside the engine's semaphore during actual trajectory execution, so
        at most n_parallel_agents ports are checked out simultaneously.

Usage:
    # Single server (default):
    env_args = {"remote_environment_url": "http://localhost:9000", "max_turns": 60}

    # Multi-server (automatic pool assignment):
    env_args = {"n_servers": 2, "base_port": 9000, "max_turns": 60}
"""

import json
import logging
import queue
import threading
import time

import requests

from rllm.environments.base.base_env import BaseEnv, UncertainExecutionError, _iter_exception_chain, is_transient_env_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Multi-server port pool
# ---------------------------------------------------------------------------
_server_port_pool: queue.Queue | None = None
_pool_init_lock = threading.Lock()


def _get_or_create_pool(n_servers: int, base_port: int) -> queue.Queue:
    """Lazily create (once) a pool of available server ports."""
    global _server_port_pool
    with _pool_init_lock:
        if _server_port_pool is None:
            _server_port_pool = queue.Queue()
            for i in range(n_servers):
                _server_port_pool.put(base_port + i)
            logger.info(f"Initialized server port pool: {[base_port + i for i in range(n_servers)]}")
    return _server_port_pool


class AppWorldEnv(BaseEnv):
    """
    AppWorld environment — pure HTTP client.

    Each instance talks to one AppWorld server for one task. No shared state
    between instances, so parallel execution across ThreadPoolExecutor threads
    is safe without locks.
    """

    def __init__(
        self,
        task=None,
        max_turns=60,
        remote_environment_url="http://localhost:9000",
        experiment_name="ace_training",
        timeout=120,
        n_servers=1,
        base_port=9000,
        **kwargs,
    ):
        """
        Initialize the AppWorld HTTP client.

        Args:
            task: Dict with at least {"task_id": "..."}.
            max_turns: Maximum number of agent turns before forced termination.
            remote_environment_url: URL of the AppWorld server.
            experiment_name: Name passed to server for logging/grouping.
            timeout: HTTP timeout in seconds for execute calls.
            n_servers: Number of AppWorld servers (for port pool).
            base_port: Starting port number (for port pool).
            **kwargs: Ignored (absorbs extra keys from from_dict).
        """
        self.task = task or {}
        self.task_id = self.task.get("task_id") or self.task.get("uid") or self.task.get("extra_info", {}).get("task_id") or ""
        self.server_url = remote_environment_url.rstrip("/")
        self.max_turns = max_turns
        self.experiment_name = experiment_name
        self.timeout = timeout

        # Multi-server config (used in reset to checkout from pool)
        self._n_servers = n_servers
        self._base_port = base_port
        self._pool_port = None  # checked-out port, returned when done

        # Mutable state (reset per episode)
        self.current_turn = 0
        self.done = False
        self._execution_in_flight = False
        self._execution_uncertainty = None

    # =========================================================================
    # BaseEnv interface
    # =========================================================================

    def reset(self):
        """
        Initialize a new task on the server.

        If multi-server mode is active (n_servers > 1), checks out a port
        from the pool FIRST. This is the correct place for checkout because
        reset() runs inside the engine's semaphore during actual trajectory
        execution, not at batch init time.

        Returns:
            (observation, info) tuple.
        """
        if self.execution_in_flight:
            raise self.mark_execution_uncertain("ExecutionStillInFlight")
        self.current_turn = 0
        self.done = False
        self._execution_uncertainty = None

        # --- Multi-server: checkout a port from the pool ---
        if self._n_servers > 1:
            # Return any previously held port (shouldn't happen, but be safe)
            self._return_port()
            pool = _get_or_create_pool(self._n_servers, self._base_port)
            # Block until a port is available. Safe because:
            # - This runs in a ThreadPoolExecutor thread (via run_in_executor)
            # - The engine semaphore limits concurrency to n_parallel_agents
            # - Pool has n_servers entries, n_parallel_agents <= n_servers
            self._pool_port = pool.get(timeout=300)  # 5min safety timeout
            self.server_url = f"http://localhost:{self._pool_port}"
            logger.info(f"Task {self.task_id}: checked out port {self._pool_port}")

        try:
            resp = self._post(
                "/initialize",
                json={
                    "task_id": self.task_id,
                    "experiment_name": self.experiment_name,
                    "load_ground_truth": True,
                    "timeout_seconds": self.timeout,
                    "raise_on_unsafe_syntax": True,
                    "null_patch_unsafe_execution": True,
                },
                timeout=self.timeout,
            )
            data = resp.json()["output"]
        except Exception as e:
            logger.error(f"Task {self.task_id}: reset failed: {e}")
            # Return port on failure so it doesn't leak
            self._return_port()
            raise

        # Map server response → agent observation format
        supervisor = data.get("supervisor", {})
        if isinstance(supervisor, dict):
            user_info = supervisor
        else:
            user_info = {
                "first_name": getattr(supervisor, "first_name", "User"),
                "last_name": getattr(supervisor, "last_name", "Test"),
                "email": getattr(supervisor, "email", "user@example.com"),
                "phone_number": getattr(supervisor, "phone_number", "+1234567890"),
            }

        observation = {
            "instruction": data.get("instruction", self.task.get("instruction", "")),
            "user_info": user_info,
        }

        app_descriptions = data.get("app_descriptions")
        if app_descriptions:
            if isinstance(app_descriptions, dict):
                observation["app_descriptions"] = json.dumps(
                    [{"name": k, "description": v} for k, v in app_descriptions.items()],
                    indent=1,
                )
            else:
                observation["app_descriptions"] = str(app_descriptions)

        info = {
            "task_id": self.task_id,
            "datetime": data.get("datetime"),
        }

        return observation, info

    def step(self, action):
        """
        Execute one step: run code, check completion, evaluate if done.

        When done=True (task completed or max turns), the port is returned
        to the pool IMMEDIATELY — before this method returns, before the
        engine's semaphore releases. This ensures the next trajectory can
        check out the freed port.

        Args:
            action: Python code string from the agent.

        Returns:
            (observation, reward, done, info) tuple.
        """
        self.current_turn += 1

        # Check max turns
        if self.current_turn >= self.max_turns:
            self.done = True
            try:
                self._post("/close", json={"task_id": self.task_id}, timeout=10)
            except Exception:
                pass
            self._return_port()
            return (
                {"error": "Reached maximum steps", "success": False},
                0.0,
                True,
                {"reason": "max_steps_reached"},
            )

        # Validate action
        if not isinstance(action, str):
            return (
                {"error": f"Invalid action type: {type(action)}. Expected string.", "success": False},
                0.0,
                False,
                {"reason": "invalid_action_type"},
            )

        # --- 1. Execute code ---
        try:
            exec_resp = self._post(
                "/execute",
                json={"task_id": self.task_id, "code": action},
                timeout=self.timeout,
            )
            output = exec_resp.json()["output"]
        except UncertainExecutionError as error:
            return self._finish_uncertain_execution(error)
        except (requests.exceptions.JSONDecodeError, KeyError, TypeError) as error:
            # A successful HTTP status without a usable output also cannot tell
            # us whether the stateful action ran successfully.
            return self._finish_uncertain_execution(self.mark_execution_uncertain(type(error).__name__))
        except Exception as e:
            logger.error(f"Task {self.task_id}: /execute error: {e}")
            return (
                {"error": str(e), "success": False},
                0.0,
                False,
                {"reason": "execution_error"},
            )

        # An outer trajectory deadline may expire while this thread is still
        # executing the request. Never continue the task after that uncertainty.
        if self._execution_uncertainty is not None:
            return self._finish_uncertain_execution(UncertainExecutionError(self._execution_uncertainty))

        # Build observation
        observation = {
            "success": True,
            "output": output,
        }

        reward = 0.0
        done = False

        # --- 2. Check completion (only if code calls complete_task) ---
        if "complete_task" in action:
            done = True
            observation["completed"] = True

            try:
                comp_resp = self._post(
                    "/task_completed",
                    json={"task_id": self.task_id},
                    timeout=30,
                )
                task_completed = comp_resp.json()["output"]

                if task_completed:
                    # --- 3. Evaluate ---
                    try:
                        eval_resp = self._post(
                            "/evaluate",
                            json={"task_id": self.task_id, "suppress_errors": True},
                            timeout=60,
                        )
                        eval_data = eval_resp.json()["output"]

                        if isinstance(eval_data, dict):
                            reward = 1.0 if eval_data.get("success") else 0.0
                        else:
                            reward = 0.0

                        logger.info(f"Task {self.task_id}: completed, reward={reward}")
                    except Exception as e:
                        logger.warning(f"Task {self.task_id}: /evaluate failed: {e}")
                        reward = 0.0
                else:
                    logger.info(f"Task {self.task_id}: complete_task called but task_completed=False")
            except Exception as e:
                logger.warning(f"Task {self.task_id}: /task_completed failed: {e}")

        self.done = done

        # Return port as soon as trajectory is done
        if done:
            # Clean up the task world on the server BEFORE returning the port,
            # so the server is ready for the next task that checks out this port.
            try:
                self._post(
                    "/close",
                    json={"task_id": self.task_id},
                    timeout=10,
                )
            except Exception:
                pass  # Server may already be shut down
            self._return_port()

        return (observation, reward, done, {"step": self.current_turn})

    def close(self):
        """
        Clean up the task world on the server and return port if still held.

        In multi-server mode, step(done=True) already calls /close and returns
        the port. So if the port is already returned, this is a no-op — avoids
        sending /close to a server that has moved on to a different task.
        """
        if self._pool_port is None and self._n_servers > 1:
            # Port already returned by step(done=True); task already cleaned up
            return

        try:
            self._post(
                "/close",
                json={"task_id": self.task_id},
                timeout=10,
            )
        except Exception:
            pass  # Server may already be shut down

        self._return_port()

    def _return_port(self):
        """Return the checked-out port to the pool (idempotent)."""
        if self._pool_port is not None and _server_port_pool is not None:
            logger.info(f"Task {self.task_id}: returning port {self._pool_port} to pool")
            _server_port_pool.put(self._pool_port)
            self._pool_port = None  # prevent double-return

    def __del__(self):
        """Safety net: return port if neither step(done) nor close() did."""
        self._return_port()

    # =========================================================================
    # Factory and capability methods
    # =========================================================================

    @staticmethod
    def from_dict(env_args: dict) -> "AppWorldEnv":
        """
        Create an AppWorldEnv from a merged {task + env_args} dict.

        NOTE: Server assignment does NOT happen here. It happens in reset(),
        which runs inside the engine's semaphore during actual trajectory
        execution. from_dict() just stores n_servers/base_port for later use.

        Args:
            env_args: Dict containing task_id, max_turns, etc.

        Returns:
            AppWorldEnv instance.
        """
        max_turns = env_args.get("max_turns", 60)
        n_servers = int(env_args.get("n_servers", 1))
        base_port = int(env_args.get("base_port", 9000))

        # For single-server mode, use the provided URL
        if n_servers <= 1:
            remote_environment_url = env_args.get("remote_environment_url") or env_args.get("remote_apis_url") or "http://localhost:9000"
        else:
            # Placeholder — will be overridden in reset() by pool checkout
            remote_environment_url = f"http://localhost:{base_port}"

        # Everything except env config keys becomes the task dict
        config_keys = {
            "max_turns",
            "remote_environment_url",
            "remote_apis_url",
            "experiment_name",
            "timeout",
            "n_servers",
            "base_port",
        }
        task = {k: v for k, v in env_args.items() if k not in config_keys}

        return AppWorldEnv(
            task=task,
            max_turns=max_turns,
            remote_environment_url=remote_environment_url,
            experiment_name=env_args.get("experiment_name", "ace_training"),
            timeout=env_args.get("timeout", 120),
            n_servers=n_servers,
            base_port=base_port,
        )

    @staticmethod
    def is_multithread_safe() -> bool:
        return True

    @staticmethod
    def is_retryable_error(exc: BaseException) -> bool:
        chain = tuple(_iter_exception_chain(exc))
        if any(isinstance(current, UncertainExecutionError) for current in chain):
            return False
        for current in chain:
            if isinstance(current, requests.exceptions.HTTPError):
                status = current.response.status_code if current.response is not None else None
                if status is not None and 500 <= status < 600:
                    return True
        return is_transient_env_error(exc)

    # =========================================================================
    # Internal helpers
    # =========================================================================

    @property
    def execution_in_flight(self):
        """Whether a sent action can still be changing server state."""
        return self._execution_in_flight

    @property
    def execution_uncertainty(self):
        return dict(self._execution_uncertainty) if self._execution_uncertainty else None

    def mark_execution_uncertain(self, error_type, http_status=None):
        """Also used when an outer deadline interrupts a threaded execute."""
        if self._execution_uncertainty is None:
            self._execution_uncertainty = {
                "termination": UncertainExecutionError.termination_reason,
                "execution_uncertain": True,
                "execution_outcome": "unknown",
                "execute_retry_attempted": False,
                "execution_error_type": error_type,
                "execution_http_status": http_status,
                "step": self.current_turn,
            }
        self.done = True
        return UncertainExecutionError(self._execution_uncertainty)

    def _finish_uncertain_execution(self, error):
        info = dict(error.info)
        logger.warning(
            "Task %s: /execute outcome uncertain at turn %s (%s, HTTP %s); ending without replay",
            self.task_id,
            self.current_turn,
            info["execution_error_type"],
            info["execution_http_status"],
        )
        self.done = True
        self.close()
        return (
            {"error": str(error), "success": False, "execution_outcome": "unknown"},
            0.0,
            True,
            info,
        )

    def _execute_once(self, url, payload, timeout):
        self._execution_in_flight = True
        try:
            response = requests.post(url, json=payload, timeout=timeout, allow_redirects=False)
            if 300 <= response.status_code < 400:
                # Redirect-following can resend a stateful POST as well.
                raise self.mark_execution_uncertain("HTTPRedirect", response.status_code)
            response.raise_for_status()
            return response
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError, requests.exceptions.ContentDecodingError) as error:
            raise self.mark_execution_uncertain(type(error).__name__) from error
        except requests.exceptions.HTTPError as error:
            status = error.response.status_code if error.response is not None else None
            if status is None or 500 <= status < 600:
                raise self.mark_execution_uncertain(type(error).__name__, status) from error
            raise
        finally:
            self._execution_in_flight = False

    def _post(self, endpoint, json=None, timeout=None):
        """Send execute once; retain bounded retries for other endpoints."""
        url = f"{self.server_url}{endpoint}"
        timeout = timeout or self.timeout
        if endpoint == "/execute":
            if self._execution_uncertainty is not None:
                raise UncertainExecutionError(self._execution_uncertainty)
            return self._execute_once(url, json, timeout)
        last_error = None

        for attempt in range(3):
            try:
                resp = requests.post(url, json=json, timeout=timeout)
                resp.raise_for_status()
                return resp
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_error = e
                if attempt < 2:
                    logger.debug(f"Task {self.task_id}: {endpoint} attempt {attempt + 1} failed ({type(e).__name__}), retrying...")
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise
            except requests.exceptions.HTTPError as e:
                last_error = e
                status = e.response.status_code if e.response is not None else None
                if status is not None and 500 <= status < 600 and attempt < 2:
                    logger.debug(f"Task {self.task_id}: {endpoint} attempt {attempt + 1} failed (HTTP {status}), retrying...")
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise

        raise last_error
