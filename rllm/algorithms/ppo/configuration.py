"""Validate opt-in PPO bindings before allocating Ray/GPU resources."""

from omegaconf import OmegaConf


def validate_ppo_config(config, *, delayed_joint: bool = False) -> None:
    """Prevent new method names from silently launching GRPO or partial resume."""
    if OmegaConf.select(config, "algorithm.adv_estimator") != "gae":
        raise ValueError("PPO requires algorithm.adv_estimator=gae; use the method's train.slurm launcher")
    if OmegaConf.select(config, "critic.enable") is False:
        raise ValueError("PPO requires an enabled learned critic")
    if OmegaConf.select(config, "rllm.workflow.use_workflow") and OmegaConf.select(config, "trainer.use_legacy_worker_impl") != "disable":
        raise ValueError("Workflow PPO requires trainer.use_legacy_worker_impl=disable for masked-padding critic loss support")
    if delayed_joint:
        # One or more complete ACE chains per task, retaining all segments.
        # These guards run in the dedicated entrypoint before Ray allocation.
        required = {
            "rllm.ace.training_mode": "joint_ppo",
            "rllm.ace.g_ref": 1,
            "rllm.ace.g_cur": 1,
            "rllm.ace.candidate_eval_rollouts": 1,
            "rllm.ace.policy_train_source": "both",
            "rllm.ace.selection_mode": "best",
            "rllm.ace.guidebook_aggregation_level": "task",
            "rllm.ace.singlemode": False,
            "rllm.ace.singlemode_curator_one_group": False,
            "rllm.rejection_sample.enable": False,
            "rllm.rejection_sample.multiplier": 1,
            "rllm.stepwise_advantage.enable": False,
            "rllm.mask_truncated_samples": False,
            "rllm.agent.overlong_filter": False,
        }
        for path, expected in required.items():
            if OmegaConf.select(config, path) != expected:
                raise ValueError(f"delayed_joint_ppo requires {path}={expected} for complete-chain PPO")
        group_size = OmegaConf.select(config, "actor_rollout_ref.rollout.n")
        if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size < 1:
            raise ValueError("delayed_joint_ppo requires actor_rollout_ref.rollout.n to be a positive integer")
    for owner in ("actor_rollout_ref.actor", "critic"):
        for operation in ("save", "load"):
            contents = OmegaConf.select(config, f"{owner}.checkpoint.{operation}_contents")
            if contents is None or not {"model", "optimizer", "extra"}.issubset(contents):
                raise ValueError(f"{owner} must {operation} model, optimizer, and extra for PPO continuation")
