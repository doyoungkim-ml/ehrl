# Setup

Launchers prepare everything they need on first use. This page explains what they
create, where it lives, and how to override it.

## GPU environment

Each launcher detects the allocated GPU family and uses `.venv-<family>` in the
checkout:

| GPU | `GPU_FAMILY` | Environment |
|---|---|---|
| A100 | `a100` | `.venv-a100` |
| H100, H200 | `hopper` (`h100`, `h200` accepted) | `.venv-hopper` |
| L40S, RTX 6000 Ada | `l40s` (`rtx6000` accepted) | `.venv-l40s` |
| RTX PRO 5000/6000 Blackwell | `rtxpro` | `.venv-rtxpro` |

Detection reads `nvidia-smi`, then the Slurm allocation. Set `GPU_FAMILY` when
neither identifies the GPU.

When the environment is missing, the launcher runs `scripts/setup_qwen35.sh` once
per GPU family, holding a lock in `.cache/locks/` so concurrent jobs wait for one
build. Setup:

1. installs `uv` for the user if it is absent;
2. fetches VeRL at the revision pinned in `compat/verl/manifest.json` into
   `.cache/verl/<revision>/` and applies the vendored patch (see
   [VeRL compatibility](verl-compatibility.md));
3. installs the pinned PyTorch, vLLM, transformers and VeRL stack;
4. builds flash-attn for the GPU architecture and applies the VeRL IPC repairs;
5. verifies the installed VeRL files and runs an import smoke test.

The flash-attn build can take hours. On clusters that cancel idle GPU jobs,
prepare the environment ahead of time on a CPU node:

```bash
GPU_FAMILY=hopper bash scripts/setup_qwen35.sh
```

An interrupted build leaves `.venv-<family>.setup-incomplete`; the next launch
rebuilds from scratch. Compute nodes need network access for the first build.

| Variable | Effect |
|---|---|
| `GPU_FAMILY` | Select the family instead of detecting it |
| `VENV_DIR` | Use or build the environment at another path |
| `PYTHON_BIN` | Use an existing interpreter; no environment is built |
| `RLLM_AUTO_SETUP=0` | Fail instead of building a missing environment |
| `RLLM_VERL_SOURCE` | Use an already patched VeRL checkout during setup |
| `RECREATE_VENV=1` | Rebuild the environment when running setup directly |

## Benchmark runtimes and data

| Benchmark | Prepared automatically | Location |
|---|---|---|
| AppWorld | Server environment, data, task IDs | `.venv_appworld_server`, `data/appworld/` (`APPWORLD_SERVER_VENV`, `APPWORLD_DATA_DIR`) |
| MBPP | Dataset download | per-job registry |
| Mastermind | Synthetic games | per-job registry |
| MiniSWEBench | SWE-Gym parquet files, container images on first use | `rllm/data/datasets/swegym`, `.cache/mini_swe_bench/` (`MINISW_DATA_DIR`, `APPTAINER_SIF_DIR`) |

MiniSWEBench has two manual prerequisites. The node needs Apptainer or Singularity
with a working `mksquashfs` (`MKSQUASHFS_BIN` selects one). The GPU environment
needs the `swegym` and `swebench` grading packages; the launcher checks
`import swegym.harness.grading` and `import swebench.harness.grading` before starting.

Existing data elsewhere is used when the corresponding variable points at it.

## W&B

Runs log to W&B under the logged-in account's default entity. Choose a team with:

```bash
export WANDB_ENTITY=<team>
```

The first job of a run records its entity, project and run ID in `run.json`;
resumed jobs reuse them and refuse a different explicit `WANDB_ENTITY`. Local W&B
files stay in the run's `wandb/` directory. `WANDB_MODE=disabled` turns logging off.

## GPU keepalive

Training and GPU evaluation jobs start a small helper that targets 85% GPU
utilization while the GPU would otherwise idle (for example during environment
steps). It backs off while the job itself uses the GPU and stops with the job.
Its readings do not guarantee the scheduler's utilization measurement or prevent
cancellation. Check both the helper log and the scheduler's device-level report
when investigating a low-utilization cancellation.

```bash
export RLLM_GPU_KEEPALIVE=0   # disable
```

`RLLM_GPU_KEEPALIVE_TARGET_UTIL` sets the target. The helper costs some training
throughput; disable it where utilization is not policed.

## Submitting jobs

Launchers carry resource defaults in their `#SBATCH` header and no account or
partition. Pass site settings as ordinary `sbatch` flags; they override the header:

```bash
PRESET=grpo MODEL_VARIANT=thinking \
  sbatch --account=<account> --partition=<partition> examples/mbpp/train/grpo/train.slurm
```

Use `bash` instead of `sbatch` inside an existing allocation, and
`RLLM_DRY_RUN=1 bash <launcher>` to print the resolved command without starting
anything. `sbatch --test-only` checks scheduler acceptance. A job imports the
source checkout while it runs, so keep the checkout unchanged until it finishes.

## CUDA toolkit

The runtime finds CUDA libraries through the selected Python environment and
passes them to training and Ray workers. A separate `nvcc` toolkit is needed only
for JIT compilation. An explicit `CUDA_HOME` selects one; otherwise packaged
toolkits, `nvcc` on `PATH` and standard locations are searched for PyTorch's CUDA
major version. `RLLM_REQUIRE_CUDA_COMPILER=1` fails early when none is found.

`python scripts/check_gpu_runtime.py --gpus 2` checks, inside an allocation, the
allocated GPUs, backend imports, NCCL all-reduce and flash-attn. It does not test
training.
