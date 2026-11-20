import subprocess
from abc import ABC, abstractmethod
from typing import Any


class UncertainExecutionError(RuntimeError):
    """A stateful action may have run; automatically replaying it is unsafe."""

    termination_reason = "ENV_EXECUTION_UNCERTAIN"

    def __init__(self, info: dict):
        self.info = dict(info)
        super().__init__("Stateful execute outcome is unknown; the action may already have run and must not be automatically replayed.")

    def __reduce__(self):
        return type(self), (self.info,)


def _iter_exception_chain(exc: BaseException):
    """Yield an exception and its explicit/implicit causes once each."""

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_transient_env_error(
    exc: BaseException,
    *,
    include_runtime_error_messages: bool = False,
) -> bool:
    """Return whether an environment exception looks retryable.

    The base class does not opt every environment into retrying arbitrary
    exceptions because data/config bugs should fail fast. Environments that
    depend on remote services, actors, containers, or subprocesses can call this
    helper from ``is_retryable_error`` to classify common transient failures
    without importing those optional dependencies globally.
    """

    exception_chain = tuple(_iter_exception_chain(exc))
    if any(isinstance(current, UncertainExecutionError) for current in exception_chain):
        return False

    # A dead shared rollout/model-serving engine is not an environment-local
    # transient. Ray can wrap this as RayTaskError(EngineDeadError), so check the
    # rendered text before the broad RayTaskError retry bucket below.
    fatal_rollout_fragments = (
        "enginedeaderror",
        "enginecore encountered an issue",
    )
    for current in exception_chain:
        text = f"{type(current).__name__}: {current}".lower()
        if any(fragment in text for fragment in fatal_rollout_fragments):
            return False

    def _matches_type(exc_types: list[type[BaseException]]) -> bool:
        return any(isinstance(current, tuple(exc_types)) for current in exception_chain)

    if _matches_type([TimeoutError, ConnectionError, subprocess.TimeoutExpired]):
        return True

    try:
        import requests

        if _matches_type(
            [
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
            ]
        ):
            return True
    except Exception:
        pass

    try:
        import urllib3

        urllib3_retryable_types = []
        for name in (
            "TimeoutError",
            "ReadTimeoutError",
            "ConnectTimeoutError",
            "ProtocolError",
        ):
            exc_type = getattr(urllib3.exceptions, name, None)
            if exc_type is not None:
                urllib3_retryable_types.append(exc_type)
        if urllib3_retryable_types and _matches_type(urllib3_retryable_types):
            return True
    except Exception:
        pass

    retryable_fragments = (
        "timed out",
        "timeout",
        "read timed out",
        "connection refused",
        "connection reset",
        "connection aborted",
        "temporarily unavailable",
        "server disconnected",
        "worker died",
        "worker crashed",
        "actor died",
        "raylet",
    )

    runtime_fragments = (
        "pod",
        "kubernetes",
        "container",
        "docker",
        "apptainer",
        "singularity",
        "image pull",
        "imagepull",
    )

    for current in exception_chain:
        text = f"{type(current).__name__}: {current}".lower()
        if any(fragment in text for fragment in retryable_fragments):
            return True
        if include_runtime_error_messages and isinstance(current, RuntimeError):
            if any(fragment in text for fragment in runtime_fragments):
                return True

    # Importing Ray can be expensive; avoid it for common HTTP/subprocess
    # failures already handled above. If a Ray exception reaches here then Ray
    # is already in use and imported by the environment, so this path is cheap.
    if any(type(current).__module__.startswith("ray.") for current in exception_chain):
        try:
            import ray

            ray_retryable_types = []
            for name in (
                "GetTimeoutError",
                "RayActorError",
                "RayTaskError",
                "TaskCancelledError",
                "WorkerCrashedError",
            ):
                exc_type = getattr(ray.exceptions, name, None)
                if exc_type is not None:
                    ray_retryable_types.append(exc_type)
            if ray_retryable_types and _matches_type(ray_retryable_types):
                return True
        except Exception:
            pass

    return False


class BaseEnv(ABC):
    @property
    def idx(self) -> Any:
        """The index or identifier of the environment, often used within a batch.

        Returns:
            The assigned index or identifier, or None if not set.
        """
        # Return the stored _idx value if it exists, otherwise return None.
        return getattr(self, "_idx", None)

    @idx.setter
    def idx(self, value: Any):
        """Set the environment index or identifier.

        This allows assigning an index or identifier (e.g., its position in a batch)
        to the environment instance after it has been created.

        Example:
            env = MyEnvSubclass()  # Assuming MyEnvSubclass inherits from BaseEnv
            env.idx = 5            # Set the index externally

        Args:
            value: The index or identifier to set for this environment.
        """
        self._idx = value

    @abstractmethod
    def reset(self) -> tuple[dict, dict]:
        """Standard Gym reset method. Resets the environment to an initial state.

        Returns:
            A tuple typically containing the initial observation and auxiliary info.
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        """Standard Gym step method. Executes one time step within the environment.

        Args:
            action: An action provided by the agent.

        Returns:
            A tuple containing (observation, reward, done, info).
        """
        pass

    def close(self):
        """Standard Gym close method. Performs any necessary cleanup."""
        return

    @staticmethod
    @abstractmethod
    def from_dict(info: dict) -> "BaseEnv":
        """Creates an environment instance from a dictionary.

        This method should be implemented by concrete subclasses to handle
        environment-specific initialization from serialized data.

        Args:
            info: A dictionary containing the necessary information to initialize the environment.

        Returns:
            An instance of the specific BaseEnv subclass.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        # BaseEnv is abstract, subclasses must implement this factory method.
        raise NotImplementedError("Subclasses must implement the 'from_dict' static method.")

    @staticmethod
    def is_multithread_safe() -> bool:
        return True

    @staticmethod
    def is_retryable_error(exc: BaseException) -> bool:
        """Whether an exception from reset/step should retry the trajectory.

        Default is deliberately conservative. Concrete remote/container-backed
        environments should override this for transient failures only.
        """

        return False
