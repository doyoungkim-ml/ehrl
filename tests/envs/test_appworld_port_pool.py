"""Stateful AppWorld servers stay exclusive across parallel task workers."""

from rllm.environments.appworld import appworld_env as module


class _Response:
    def json(self):
        return {"output": {"instruction": "test", "supervisor": {}}}


def test_sixteen_trajectories_checkout_distinct_servers(monkeypatch):
    monkeypatch.setattr(module, "_server_port_pool", None)
    monkeypatch.setattr(module.AppWorldEnv, "_post", lambda self, *args, **kwargs: _Response())
    envs = [
        module.AppWorldEnv(task={"task_id": str(index)}, n_servers=16, base_port=12000)
        for index in range(16)
    ]
    for env in envs:
        env.reset()
    assert {env._pool_port for env in envs} == set(range(12000, 12016))
    for env in envs:
        env.close()
    assert module._server_port_pool.qsize() == 16
