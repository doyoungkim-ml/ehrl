#!/usr/bin/env python3
"""Publish a block-FP8 copy of one BF16 Hugging Face Qwen checkpoint for vLLM.

The weights are quantized exactly as VeRL's FP8 rollout quantizes the BF16
actor at every weight sync (actor_rollout_ref.rollout.quantization=fp8):

* every dense decoder Linear weight (q/k/v/o_proj, gate/up/down_proj) is cast
  to bfloat16, split into 128x128 blocks, and stored as float8_e4m3fn with a
  float32 ``<name>_scale_inv`` (block absmax / 448) of shape
  ceil(out/128) x ceil(in/128); partial edge blocks are zero-padded;
* embeddings, lm_head, norms and biases keep their original tensors;
* config.json gains VeRL's injected quantization_config (dynamic e4m3
  activations, weight_block_size [128, 128], the MoE ``mlp.gate`` ignore list).

``--kernel verl`` (default) quantizes with the function the rollout weight
sync calls, ``scaled_fp8_blockwise`` from the installed (or ``--verl-source``)
``verl/utils/kernel/fp8_kernel.py``, on the visible CUDA device with Triton. It
fails when no CUDA device or Triton kernel is available. Triton can lower the
kernel's float32 -> e4m3 cast differently by GPU generation (through float16,
expected on compute capability < 9.0), so the recorded ``compute_capability`` matters;
``--min-capability 9.0`` refuses to convert on GPUs older than the H100/H200
training rollouts. ``--kernel replica`` is the explicit CPU fallback: the same
arithmetic in float32 with a direct round-to-nearest-even e4m3 cast
(scale = max(absmax, 1e-10) / 448, q = clamp(x * (1 / scale), -448, 448)); it
can differ from the GPU kernel by one e4m3 code on a small fraction of elements.

``--verify-against-verl`` re-quantizes every tensor with VeRL's kernel on the
visible device and reports code and scale mismatches; it fails only when the
target is expected to be bit-identical (made by the kernel on a GPU of the same
compute capability). The target is never overwritten; a completed target is
reused only when ``fp8_conversion.json`` matches the source, converter and
quantizer, and it records the implementation and device used.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import struct
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

FORMAT_VERSION = 1
MARKER = "fp8_conversion.json"
BLOCK = (128, 128)
FP8_MAX = 448.0
EPS = 1e-10
SUPPORTED_MODEL_TYPES = {"qwen2", "qwen3"}
QUANTIZED = re.compile(r"^model\.layers\.(\d+)\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$")
# Tensors that VeRL's rollout leaves unquantized: vLLM's FP8 method covers
# LinearBase layers only (not embeddings/lm_head); norms and biases are not
# Linear weights.
KEPT = re.compile(
    r"^(model\.embed_tokens\.weight|lm_head\.weight|model\.norm\.weight|model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight|model\.layers\.\d+\.self_attn\.(q_norm|k_norm)\.weight|model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj\.bias)$"
)
SKIPPED_FILES = {"merge_complete.json", "model.safetensors.index.json", MARKER}


class ConversionError(RuntimeError):
    pass


def quantization_config(num_hidden_layers: int) -> dict:
    """The quantization_config VeRL injects through vLLM hf_overrides for FP8 rollout."""
    return {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": list(BLOCK),
        "ignored_layers": [f"model.layers.{layer}.mlp.gate" for layer in range(num_hidden_layers)],
    }


def quantize_block_fp8(weight, block=BLOCK):
    """Return (float8_e4m3fn weight, float32 scale_inv) for one 2-D tensor."""
    import torch

    if weight.dim() != 2:
        raise ConversionError(f"Block FP8 quantization needs a 2-D tensor, got shape {tuple(weight.shape)}")
    rows, columns = weight.shape
    block_rows, block_columns = block
    # VeRL casts each synced weight to the rollout dtype before quantizing.
    data = weight.to(torch.bfloat16).to(torch.float32)
    pad_rows, pad_columns = (-rows) % block_rows, (-columns) % block_columns
    if pad_rows or pad_columns:
        data = torch.nn.functional.pad(data, (0, pad_columns, 0, pad_rows))
    grid_rows, grid_columns = data.shape[0] // block_rows, data.shape[1] // block_columns
    blocks = data.reshape(grid_rows, block_rows, grid_columns, block_columns)
    absmax = blocks.abs().amax(dim=(1, 3)).clamp_min(EPS)
    scale = absmax / FP8_MAX
    inverse = 1.0 / scale
    quantized = (blocks * inverse[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    quantized = quantized.reshape(data.shape)[:rows, :columns].contiguous()
    return quantized, scale.to(torch.float32).contiguous()


def dequantize_block_fp8(quantized, scale_inv, block=BLOCK):
    import torch

    rows, columns = quantized.shape
    expanded = scale_inv.repeat_interleave(block[0], dim=0).repeat_interleave(block[1], dim=1)[:rows, :columns]
    return quantized.to(torch.float32) * expanded


def regular_file(path: Path) -> os.stat_result:
    # Hugging Face hub snapshots are symlinks into the blob store; follow them.
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ConversionError(f"Expected a regular file: {path}")
    return info


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safetensors_header_digest(path: Path) -> str:
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if length <= 1 or length > 100 * 1024 * 1024:
            raise ConversionError(f"Invalid safetensors header: {path}")
        return hashlib.sha256(stream.read(length)).hexdigest()


def weight_files(source: Path) -> list[str]:
    index = source / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text()).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ConversionError(f"Invalid safetensors index: {index}")
        names = sorted(set(weight_map.values()))
    else:
        names = ["model.safetensors"]
    for name in names:
        if Path(name).name != name or not (source / name).is_file():
            raise ConversionError(f"Missing safetensors shard {name!r} in {source}")
    return names


def source_identity(source: Path) -> dict:
    config = source / "config.json"
    files = {}
    for name in weight_files(source):
        info = regular_file(source / name)
        files[name] = {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns, "header_sha256": safetensors_header_digest(source / name)}
    return {"path": str(source), "config_sha256": sha256(config), "weight_files": files}


def converter_identity() -> dict:
    return {"converter_sha256": sha256(Path(__file__).resolve()), "format_version": FORMAT_VERSION}


def verl_kernel_path(verl_source: Path | None) -> Path:
    """The fp8_kernel.py the rollout imports, located without importing VeRL."""
    if verl_source is not None:
        path = verl_source / "verl/utils/kernel/fp8_kernel.py"
    else:
        import importlib.util

        spec = importlib.util.find_spec("verl")
        if spec is None or not spec.origin:
            raise ConversionError("VeRL is not installed in this interpreter; pass --verl-source or use --kernel replica explicitly")
        path = Path(spec.origin).parent / "utils/kernel/fp8_kernel.py"
    if not path.is_file():
        raise ConversionError(f"Missing VeRL FP8 kernel: {path}")
    return path


def load_verl_kernel(path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("verl_fp8_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def requested_quantizer(kernel: str, verl_source: Path | None) -> dict:
    """The reuse identity of the requested quantizer (no CUDA needed)."""
    if kernel == "replica":
        return {"implementation": "cpu-replica", "kernel_sha256": None}
    if kernel != "verl":
        raise ConversionError(f"Unknown kernel {kernel!r}; use verl or replica")
    return {"implementation": "verl-triton", "kernel_sha256": sha256(verl_kernel_path(verl_source))}


def capability_at_least(value: str | None, minimum: str | None) -> bool:
    if not minimum:
        return True
    if not value:
        return False
    return tuple(int(part) for part in value.split(".")) >= tuple(int(part) for part in minimum.split("."))


def kernel_device():
    import torch

    return torch.device("cuda", torch.cuda.current_device())


def select_quantizer(kernel: str, verl_source: Path | None, min_capability: str | None):
    """Return (quantize(weight) -> (fp8, scale_inv) on CPU, metadata)."""
    import torch

    identity = requested_quantizer(kernel, verl_source)
    if kernel == "replica":
        return quantize_block_fp8, {**identity, "device": "cpu", "compute_capability": None, "torch_version": torch.__version__}
    if not torch.cuda.is_available():
        raise ConversionError("--kernel verl needs a visible CUDA device; run inside the GPU allocation or choose --kernel replica explicitly")
    path = verl_kernel_path(verl_source)
    module = load_verl_kernel(path)
    if not module.is_triton_available() or getattr(module, "_DISABLE_TRITON_FP8", False):
        raise ConversionError("VeRL's Triton FP8 kernel is unavailable (Triton missing or VERL_DISABLE_TRITON_FP8 set); refusing its PyTorch fallback")
    device = kernel_device()
    major, minor = torch.cuda.get_device_capability(device)
    capability = f"{major}.{minor}"
    if not capability_at_least(capability, min_capability):
        raise ConversionError(f"GPU compute capability {capability} is below --min-capability {min_capability}; Triton rounds the FP8 cast differently there than on the training rollout GPUs")
    if major < 9:
        print(f"[fp8] warning: compute capability {capability} < 9.0; Triton is expected to cast through float16 here, unlike H100/H200 rollouts", file=sys.stderr)
    import triton

    def quantize(weight):
        # verl.utils.vllm.vllm_fp8_utils.quant_weights: scaled_fp8_blockwise(v.to(dtype), block), scale.squeeze(-1)
        quantized, scale = module.scaled_fp8_blockwise(weight.to(device=device, dtype=torch.bfloat16), list(BLOCK))
        return quantized.cpu().contiguous(), scale.squeeze(-1).to(torch.float32).cpu().contiguous()

    metadata = {
        **identity,
        "kernel_path": str(path),
        "device": torch.cuda.get_device_name(device),
        "compute_capability": capability,
        "triton_version": triton.__version__,
        "torch_version": torch.__version__,
    }
    return quantize, metadata


def resolve_source(source_dir: str | None, hub_model: str | None) -> Path:
    if bool(source_dir) == bool(hub_model):
        raise ConversionError("Pass exactly one of --source-dir or --hub-model")
    if source_dir:
        return Path(source_dir).expanduser().resolve(strict=True)
    from huggingface_hub import snapshot_download

    # The BF16 base snapshot is the normal cached model; no FP8 download is needed.
    return Path(snapshot_download(hub_model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "tokenizer*", "merges.txt", "vocab.json"])).resolve()


def read_config(source: Path) -> dict:
    config = json.loads((source / "config.json").read_text())
    if not isinstance(config, dict):
        raise ConversionError(f"config.json is not an object: {source}")
    if config.get("quantization_config"):
        raise ConversionError(f"Source is already quantized ({config['quantization_config'].get('quant_method')}): {source}")
    if config.get("model_type") not in SUPPORTED_MODEL_TYPES:
        raise ConversionError(f"Unsupported model_type {config.get('model_type')!r}; supported: {sorted(SUPPORTED_MODEL_TYPES)}")
    if not isinstance(config.get("num_hidden_layers"), int) or config["num_hidden_layers"] <= 0:
        raise ConversionError("config.json needs a positive num_hidden_layers")
    return config


def convert_tensors(tensors: dict, layer_count: int, quantize=quantize_block_fp8) -> tuple[dict, list[str]]:
    output, quantized = {}, []
    for name, tensor in tensors.items():
        match = QUANTIZED.match(name)
        if match:
            if int(match.group(1)) >= layer_count:
                raise ConversionError(f"Tensor {name} exceeds num_hidden_layers={layer_count}")
            output[name], output[name + "_scale_inv"] = quantize(tensor)
            quantized.append(name)
        elif KEPT.match(name):
            output[name] = tensor
        else:
            # A Linear weight left in BF16 here would be FP8 in the rollout.
            raise ConversionError(f"Unrecognized tensor {name}; refusing to guess whether VeRL's FP8 rollout quantizes it")
    return output, quantized


def build(source: Path, staging: Path, quantize=quantize_block_fp8) -> dict:
    from safetensors import safe_open
    from safetensors.torch import save_file

    config = read_config(source)
    layer_count = config["num_hidden_layers"]
    quantized_names, weight_map = [], {}
    for name in weight_files(source):
        with safe_open(str(source / name), framework="pt") as stream:
            tensors = {key: stream.get_tensor(key) for key in stream.keys()}
        converted, quantized = convert_tensors(tensors, layer_count, quantize)
        quantized_names.extend(quantized)
        save_file(converted, str(staging / name), metadata={"format": "pt"})
        weight_map.update({key: name for key in converted})
    expected = {
        f"model.layers.{layer}.{module}.weight"
        for layer in range(layer_count)
        for module in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    }
    if set(quantized_names) != expected:
        missing = sorted(expected - set(quantized_names))[:4]
        raise ConversionError(f"Quantized Linear weights do not cover every decoder layer; missing e.g. {missing}")
    if (source / "model.safetensors.index.json").exists():
        (staging / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": dict(sorted(weight_map.items()))}, indent=2) + "\n")
    config["quantization_config"] = quantization_config(layer_count)
    (staging / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    for path in sorted(source.iterdir()):
        if path.is_file() and path.suffix != ".safetensors" and path.name not in SKIPPED_FILES and path.name != "config.json":
            shutil.copyfile(path, staging / path.name)
    return {"quantized_tensors": len(quantized_names), "quantization_config": config["quantization_config"]}


@contextlib.contextmanager
def target_lock(target: Path):
    import fcntl

    lock_path = target.with_name(target.name + ".fp8.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ConversionError(f"Another converter owns {lock_path}") from exc
        yield
    finally:
        os.close(descriptor)


def publish(staging: Path, target: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from merge_eval_checkpoint import rename_noreplace

    try:
        rename_noreplace(staging, target)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            raise ConversionError(f"Target appeared during conversion: {target}") from exc
        if exc.errno not in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
            raise
        # Without renameat2 the lock still excludes cooperating converters.
        if target.exists() or target.is_symlink():
            raise ConversionError(f"Target appeared during conversion: {target}") from exc
        os.rename(staging, target)


def convert(source: Path, target: Path, *, kernel: str = "verl", verl_source: Path | None = None, min_capability: str | None = None) -> dict:
    target = target.expanduser().absolute()
    target = target.parent.resolve() / target.name
    if target == source or target.is_relative_to(source) or source.is_relative_to(target):
        raise ConversionError("Source and target must be separate, non-nested directories")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target_lock(target):
        identity = {"source": source_identity(source), **converter_identity()}
        wanted = requested_quantizer(kernel, verl_source)
        if target.exists() or target.is_symlink():
            marker = target / MARKER
            if not marker.is_file():
                raise ConversionError(f"Existing target is not a completed FP8 conversion; it will not be overwritten: {target}")
            recorded = json.loads(marker.read_text())
            if {key: recorded.get(key) for key in identity} != identity:
                raise ConversionError(f"Existing FP8 target was made from another source or converter: {target}")
            made_by = recorded.get("quantizer") or {}
            if {key: made_by.get(key) for key in wanted} != wanted:
                raise ConversionError(f"Existing FP8 target was made by {made_by.get('implementation')!r}, not the requested {wanted['implementation']!r}: {target}")
            if not capability_at_least(made_by.get("compute_capability"), min_capability):
                raise ConversionError(f"Existing FP8 target was made on compute capability {made_by.get('compute_capability')}, below {min_capability}: {target}")
            return {"status": "reused", "target_dir": str(target), "quantizer": made_by}
        quantize, quantizer = select_quantizer(kernel, verl_source, min_capability)
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.fp8-", dir=target.parent))
        try:
            summary = {**build(source, staging, quantize), "quantizer": quantizer}
            if source_identity(source) != identity["source"]:
                raise ConversionError("Source changed during conversion")
            (staging / MARKER).write_text(json.dumps({**identity, **summary, "completed_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
            publish(staging, target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return {"status": "converted", "target_dir": str(target), **summary}


def verify_against_verl(target: Path, source: Path, verl_source: Path | None) -> dict:
    """Compare every quantized tensor with VeRL's own blockwise kernel."""
    import torch
    from safetensors import safe_open

    kernel = load_verl_kernel(verl_kernel_path(verl_source))
    use_gpu = torch.cuda.is_available()
    device = kernel_device() if use_gpu else torch.device("cpu")
    implementation = "triton" if use_gpu and kernel.is_triton_available() and not getattr(kernel, "_DISABLE_TRITON_FP8", False) else "pytorch"
    capability = "{}.{}".format(*torch.cuda.get_device_capability(device)) if use_gpu else None
    made_by = json.loads((target / MARKER).read_text()).get("quantizer") or {}
    # Bit-identity is expected only for kernel output on the same GPU generation.
    expected_identical = implementation == "triton" and made_by.get("implementation") == "verl-triton" and made_by.get("compute_capability") == capability
    compared = mismatched = scale_mismatched = max_code_difference = 0
    for name in weight_files(source):
        with safe_open(str(source / name), framework="pt") as original, safe_open(str(target / name), framework="pt") as converted:
            for key in original.keys():
                if not QUANTIZED.match(key):
                    continue
                if implementation == "triton":
                    expected, expected_scale = kernel.scaled_fp8_blockwise(original.get_tensor(key).to(device, torch.bfloat16), list(BLOCK))
                else:
                    expected, expected_scale = kernel._scaled_fp8_blockwise_pytorch(original.get_tensor(key).to(torch.bfloat16), list(BLOCK))
                actual = converted.get_tensor(key).view(torch.uint8).to(device).int()
                reference = expected.view(torch.uint8).to(device).int()
                scale = converted.get_tensor(key + "_scale_inv").to(device)
                expected_scale = expected_scale.reshape(scale.shape).to(device=device, dtype=torch.float32)
                compared += actual.numel()
                different = actual != reference
                mismatched += int(different.sum())
                if different.any():
                    # Same-sign codes are ordered; a difference of 1 is the adjacent e4m3 value.
                    max_code_difference = max(max_code_difference, int((actual - reference)[different].abs().max()))
                scale_mismatched += int((scale.view(torch.int32) != expected_scale.contiguous().view(torch.int32)).sum())
                if not torch.allclose(scale, expected_scale, rtol=1e-6, atol=0):
                    raise ConversionError(f"Scale mismatch beyond 1e-6 against VeRL's {implementation} kernel for {key}")
    return {
        "implementation": implementation,
        "device": "cuda" if use_gpu else "cpu",
        "compute_capability": capability,
        "target_quantizer": made_by.get("implementation"),
        "target_compute_capability": made_by.get("compute_capability"),
        "expected_identical": expected_identical,
        "compared_elements": compared,
        "mismatched_elements": mismatched,
        "max_code_difference": max_code_difference,
        "scale_bitwise_mismatches": scale_mismatched,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", help="Merged BF16 Hugging Face checkpoint directory")
    parser.add_argument("--hub-model", help="Hugging Face model id of a BF16 base model (uses the local cache when offline)")
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument("--kernel", choices=("verl", "replica"), default="verl", help="verl: VeRL's Triton kernel on the visible GPU (default); replica: explicit CPU fallback")
    parser.add_argument("--verl-source", type=Path, help="VeRL root containing verl/utils/kernel/fp8_kernel.py (default: the installed package)")
    parser.add_argument("--min-capability", help="Refuse kernel conversion below this CUDA compute capability, e.g. 9.0")
    parser.add_argument("--verify-against-verl", nargs="?", const="installed", metavar="VERL_SOURCE", help="Re-quantize with VeRL's kernel (Triton on CUDA) and compare with the target")
    args = parser.parse_args(argv)
    try:
        source = resolve_source(args.source_dir, args.hub_model)
        verl_source = args.verl_source.resolve() if args.verl_source else None
        result = convert(source, args.target_dir, kernel=args.kernel, verl_source=verl_source, min_capability=args.min_capability)
        if args.verify_against_verl:
            verify_source = verl_source if args.verify_against_verl == "installed" else Path(args.verify_against_verl).resolve()
            result["verification"] = verify_against_verl(Path(result["target_dir"]), source, verify_source)
            verification = result["verification"]
            if verification["expected_identical"] and (verification["mismatched_elements"] or verification["scale_bitwise_mismatches"]):
                print(json.dumps(result, sort_keys=True))
                return 3
    except (ConversionError, OSError, ValueError, KeyError) as exc:
        print(f"[fp8] {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
