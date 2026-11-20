import os

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # TODO: disable compile cache due to cache corruption issue
        # https://github.com/vllm-project/vllm/issues/31199
        "VLLM_DISABLE_COMPILE_CACHE": "1",
        # To prevent hanging or crash during synchronization of weights between actor and rollout
        # in disaggregated mode. See:
        # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
        # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
        "NCCL_CUMEM_ENABLE": "0",
    },
}

FORWARD_PREFIXES = [
    "VLLM_",
    "SGL_",
    "SGLANG_",
    "HF_",
    "TOKENIZERS_",
    "DATASETS_",
    "TORCH_",
    "PYTORCH_",
    "DEEPSPEED_",
    "MEGATRON_",
    "NCCL_",
    "CUDA_",
    "CUBLAS_",
    "CUDNN_",
    "NV_",
    "NVIDIA_",
]

FORWARD_EXACT_VARS = [
    "WANDB_ENTITY",
    "WANDB_PROJECT",
    "WANDB_RUN_ID",
    "WANDB_RESUME",
    "WANDB_DIR",
    "WANDB_MODE",
    "RLLM_RUN_DIR",
    "CURL_CA_BUNDLE",
    "FLASHINFER_WORKSPACE_BASE",
    "LD_LIBRARY_PATH",
    "PATH",
    "REQUESTS_CA_BUNDLE",
    "RLLM_RUNTIME_LD_LIBRARY_PATH",
    "RLLM_RUNTIME_DIR",
    "RLLM_CACHE_DIR",
    "RLLM_DIR",
    "SSL_CERT_FILE",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "VIRTUAL_ENV",
    "XDG_CACHE_HOME",
    "RAY_TMPDIR",
    "RLLM_TMPDIR",
    "TEMP",
    "TMP",
    "TMPDIR",
]


def _get_forwarded_env_vars():
    """
    Get the forwarded environment variables. The `RLLM_EXCLUDE` environment variable can be used to
    exclude specific environment variables or all variables with a specific prefix.

    Example:
    ```
    RLLM_EXCLUDE=VLLM*,CUDA*,NCCL_IB_DISABLE
    ```
    will exclude all variables with prefix `VLLM_`, `CUDA_`, and `NCCL_IB_DISABLE`.

    By default, all environment variables with prefix in `FORWARD_PREFIXES` are forwarded.
    """
    if os.environ.get("RLLM_EXCLUDE", None) is not None:
        rllm_exclude = str(os.environ.get("RLLM_EXCLUDE")).split(",")
    else:
        rllm_exclude = []

    forward_prefix = FORWARD_PREFIXES.copy()

    exclude_vars = set()
    for name in rllm_exclude:
        if "*" in name:  # denote a prefix match, e.g. "VLLM*"
            forward_prefix.remove(name.replace("*", "_"))
        else:
            exclude_vars.add(name)

    forwarded = {k: v for k, v in os.environ.items() if (any(k.startswith(p) for p in forward_prefix) or k in FORWARD_EXACT_VARS) and k not in exclude_vars}
    return forwarded


def get_ppo_ray_runtime_env():
    env = PPO_RAY_RUNTIME_ENV.get("env_vars", {}).copy()
    env.update(_get_forwarded_env_vars())
    return {"env_vars": env}
