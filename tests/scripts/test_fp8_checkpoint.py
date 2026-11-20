"""Block-FP8 evaluation checkpoints match VeRL's FP8 rollout quantization."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

ROOT = Path(__file__).resolve().parents[2]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


converter = load(ROOT / "scripts/convert_fp8_checkpoint.py", "convert_fp8_checkpoint")
merge = load(ROOT / "scripts/merge_eval_checkpoint.py", "merge_eval_checkpoint_for_fp8")


@pytest.fixture(scope="module")
def verl_kernel():
    source = os.environ.get("VERL_RESUME_SOURCE")
    if not source:
        pytest.skip("Set VERL_RESUME_SOURCE to compare with VeRL's FP8 kernel")
    return load(Path(source) / "verl/utils/kernel/fp8_kernel.py", "verl_fp8_kernel_for_tests")


def triton_formula(weight):
    """Element-wise restatement of VeRL's Triton kernel, one block at a time."""
    data = weight.to(torch.bfloat16).float()
    rows, columns = data.shape
    out = torch.empty(rows, columns, dtype=torch.float8_e4m3fn)
    scales = torch.empty((rows + 127) // 128, (columns + 127) // 128)
    for i in range(scales.shape[0]):
        for j in range(scales.shape[1]):
            block = data[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128]
            scale = torch.clamp(block.abs().max(), min=1e-10) / 448.0
            out[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128] = torch.clamp(block * (1.0 / scale), -448.0, 448.0).to(torch.float8_e4m3fn)
            scales[i, j] = scale
    return out, scales


@pytest.mark.parametrize("shape", [(256, 128), (1024, 384), (200, 300), (130, 257), (64, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_quantizer_matches_verl_and_round_trips(verl_kernel, shape, dtype):
    generator = torch.Generator().manual_seed(sum(shape))
    weight = (torch.randn(shape, generator=generator) * 0.02).to(dtype)
    weight[0, :7] = torch.tensor([3.0, -2.5, 1e-6, 0.0, 0.5, -0.5, 7.0])  # outliers and tiny values
    quantized, scale = converter.quantize_block_fp8(weight)
    assert quantized.dtype == torch.float8_e4m3fn and quantized.shape == shape
    assert scale.dtype == torch.float32 and scale.shape == ((shape[0] + 127) // 128, (shape[1] + 127) // 128)

    expected, expected_scale = triton_formula(weight)
    assert torch.equal(quantized.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(scale, expected_scale)

    # VeRL's non-Triton fallback multiplies by 448/absmax instead of 1/(absmax/448);
    # that float32 rounding moves rare elements to the adjacent e4m3 code.
    verl_q, verl_descale = verl_kernel._scaled_fp8_blockwise_pytorch(weight.to(torch.bfloat16), [128, 128])
    torch.testing.assert_close(scale, verl_descale.reshape(scale.shape), rtol=1e-6, atol=0)
    ours, theirs = quantized.view(torch.uint8).int(), verl_q.view(torch.uint8).int()
    different = ours != theirs
    assert int(different.sum()) <= max(1, quantized.numel() // 1000)
    assert ((ours[different] - theirs[different]).abs() == 1).all() and ((ours[different] ^ theirs[different]) < 128).all()

    restored = converter.dequantize_block_fp8(quantized, scale)
    original = weight.to(torch.bfloat16).float()
    verl_restored = verl_q.float() * verl_descale.reshape(scale.shape).repeat_interleave(128, 0).repeat_interleave(128, 1)[: shape[0], : shape[1]]
    torch.testing.assert_close(restored[~different], verl_restored[~different], rtol=1e-6, atol=0)
    # e4m3 keeps three mantissa bits: relative error <= 2^-4 for normal values.
    big = original.abs() > scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[: shape[0], : shape[1]] * 2**-6
    assert ((restored - original).abs()[big] <= original.abs()[big] * 2**-4 + 1e-12).all()


def tiny_model(path: Path, hidden=200, intermediate=300, layers=2, sharded=False, extra=None):
    path.mkdir(parents=True)
    generator = torch.Generator().manual_seed(0)
    tensors = {"model.embed_tokens.weight": torch.randn(50, hidden, generator=generator), "model.norm.weight": torch.ones(hidden), "lm_head.weight": torch.randn(50, hidden, generator=generator)}
    for layer in range(layers):
        prefix = f"model.layers.{layer}."
        tensors.update(
            {
                prefix + "input_layernorm.weight": torch.ones(hidden),
                prefix + "post_attention_layernorm.weight": torch.ones(hidden),
                prefix + "self_attn.q_norm.weight": torch.ones(64),
                prefix + "self_attn.k_norm.weight": torch.ones(64),
                prefix + "self_attn.q_proj.weight": torch.randn(256, hidden, generator=generator),
                prefix + "self_attn.k_proj.weight": torch.randn(128, hidden, generator=generator),
                prefix + "self_attn.v_proj.weight": torch.randn(128, hidden, generator=generator),
                prefix + "self_attn.o_proj.weight": torch.randn(hidden, 256, generator=generator),
                prefix + "mlp.gate_proj.weight": torch.randn(intermediate, hidden, generator=generator),
                prefix + "mlp.up_proj.weight": torch.randn(intermediate, hidden, generator=generator),
                prefix + "mlp.down_proj.weight": torch.randn(hidden, intermediate, generator=generator),
            }
        )
    tensors.update(extra or {})
    if sharded:
        names = sorted(tensors)
        shards = {"model-00001-of-00002.safetensors": names[: len(names) // 2], "model-00002-of-00002.safetensors": names[len(names) // 2 :]}
        for shard, keys in shards.items():
            safetensors_torch.save_file({key: tensors[key] for key in keys}, str(path / shard))
        (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": {key: shard for shard, keys in shards.items() for key in keys}}))
    else:
        safetensors_torch.save_file(tensors, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps({"model_type": "qwen3", "num_hidden_layers": layers, "torch_dtype": "bfloat16", "tie_word_embeddings": False}))
    (path / "tokenizer_config.json").write_text("{}")
    (path / "tokenizer.json").write_text("{}")
    (path / "generation_config.json").write_text("{}")
    (path / "merge_complete.json").write_text("{}")
    return tensors


def read_all(path: Path) -> dict:
    from safetensors import safe_open

    tensors = {}
    for name in path.glob("*.safetensors"):
        with safe_open(str(name), framework="pt") as stream:
            tensors.update({key: stream.get_tensor(key) for key in stream.keys()})
    return tensors


@pytest.mark.parametrize("sharded", [False, True])
def test_conversion_publishes_vllm_block_fp8_layout(tmp_path, sharded):
    source_tensors = tiny_model(tmp_path / "bf16", sharded=sharded)
    result = converter.main(["--source-dir", str(tmp_path / "bf16"), "--target-dir", str(tmp_path / "fp8"), "--kernel", "replica"])
    assert result == 0
    target = tmp_path / "fp8"
    config = json.loads((target / "config.json").read_text())
    assert config["quantization_config"] == {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "ignored_layers": ["model.layers.0.mlp.gate", "model.layers.1.mlp.gate"],
    }
    tensors = read_all(target)
    for name, original in source_tensors.items():
        if converter.QUANTIZED.match(name):
            assert tensors[name].dtype == torch.float8_e4m3fn
            assert tensors[name + "_scale_inv"].shape == ((original.shape[0] + 127) // 128, (original.shape[1] + 127) // 128)
            expected, expected_scale = converter.quantize_block_fp8(original)
            assert torch.equal(tensors[name].view(torch.uint8), expected.view(torch.uint8))
            assert torch.equal(tensors[name + "_scale_inv"], expected_scale)
        else:
            assert torch.equal(tensors[name], original), name
    assert len(tensors) == len(source_tensors) + 14
    assert not (target / "merge_complete.json").exists()
    assert {"tokenizer.json", "tokenizer_config.json", "generation_config.json", "fp8_conversion.json"} <= {path.name for path in target.iterdir()}
    marker = json.loads((target / "fp8_conversion.json").read_text())
    assert marker["quantized_tensors"] == 14 and marker["source"]["path"] == str((tmp_path / "bf16").resolve())
    # The merged-checkpoint validator accepts the FP8 payload and its index.
    snapshot = merge.output_snapshot(target)
    assert snapshot["weight_files"]
    # Rerunning reuses the identical conversion.
    assert converter.convert((tmp_path / "bf16").resolve(), target, kernel="replica")["status"] == "reused"


def test_existing_target_from_other_source_or_partial_output_is_never_replaced(tmp_path):
    tiny_model(tmp_path / "a")
    tiny_model(tmp_path / "b", hidden=128)
    assert converter.main(["--source-dir", str(tmp_path / "a"), "--target-dir", str(tmp_path / "fp8"), "--kernel", "replica"]) == 0
    with pytest.raises(converter.ConversionError, match="another source"):
        converter.convert((tmp_path / "b").resolve(), tmp_path / "fp8", kernel="replica")
    (tmp_path / "partial").mkdir()
    with pytest.raises(converter.ConversionError, match="not a completed"):
        converter.convert((tmp_path / "a").resolve(), tmp_path / "partial", kernel="replica")
    assert not any(path.name.startswith(".fp8-") for path in tmp_path.iterdir())


@pytest.mark.parametrize(
    "extra,config,message",
    [
        ({"model.layers.0.mlp.experts.0.up_proj.weight": torch.zeros(128, 128)}, {}, "Unrecognized tensor"),
        ({}, {"model_type": "llama"}, "Unsupported model_type"),
        ({}, {"quantization_config": {"quant_method": "fp8"}}, "already quantized"),
    ],
)
def test_unsupported_sources_fail_without_output(tmp_path, extra, config, message):
    tiny_model(tmp_path / "bf16", extra=extra)
    if config:
        path = tmp_path / "bf16" / "config.json"
        path.write_text(json.dumps({**json.loads(path.read_text()), **config}))
    with pytest.raises(converter.ConversionError, match=message):
        converter.convert((tmp_path / "bf16").resolve(), tmp_path / "fp8", kernel="replica")
    assert not (tmp_path / "fp8").exists()


def test_cli_verifies_against_verl_kernel(tmp_path, verl_kernel):
    tiny_model(tmp_path / "bf16")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/convert_fp8_checkpoint.py"),
            "--source-dir",
            str(tmp_path / "bf16"),
            "--target-dir",
            str(tmp_path / "fp8"),
            "--kernel",
            "replica",
            "--verify-against-verl",
            os.environ["VERL_RESUME_SOURCE"],
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    verification = json.loads(result.stdout)["verification"]
    assert verification["implementation"] == "pytorch" and verification["compared_elements"] > 0
    assert verification["target_quantizer"] == "cpu-replica" and verification["expected_identical"] is False
    assert verification["max_code_difference"] <= 1
    assert verification["target_quantizer"] == "cpu-replica" and verification["expected_identical"] is False
    assert verification["max_code_difference"] <= 1
    assert verification["mismatched_elements"] <= verification["compared_elements"] // 1000


class FakeVerlKernel:
    """Stands in for verl/utils/kernel/fp8_kernel.py; records the rollout call."""

    _DISABLE_TRITON_FP8 = False

    def __init__(self, triton=True):
        self.triton, self.calls = triton, []

    def is_triton_available(self):
        return self.triton

    def scaled_fp8_blockwise(self, data, weight_block_size):
        self.calls.append((data.dtype, tuple(data.shape), list(weight_block_size)))
        quantized, scale = converter.quantize_block_fp8(data)
        # Distinguish kernel output from the CPU replica in the published files.
        return quantized, (scale * 2).unsqueeze(-1)


@pytest.fixture
def fake_gpu(monkeypatch, tmp_path):
    """CUDA device of the given capability with VeRL's kernel; computation stays on the CPU."""
    import types

    kernel_file = tmp_path / "verl-src/verl/utils/kernel/fp8_kernel.py"
    kernel_file.parent.mkdir(parents=True)
    kernel_file.write_text("# kernel source identity\n")
    state = {"kernel": FakeVerlKernel(), "capability": (9, 0)}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: state["capability"])
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "Fake H200")
    monkeypatch.setattr(converter, "kernel_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(converter, "load_verl_kernel", lambda path: state["kernel"])
    monkeypatch.setitem(sys.modules, "triton", types.SimpleNamespace(__version__="3.6.0"))
    state["verl_source"] = tmp_path / "verl-src"
    return state


def test_default_kernel_is_verl_and_fails_loudly_without_a_gpu(tmp_path, monkeypatch):
    tiny_model(tmp_path / "bf16")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    kernel_file = tmp_path / "verl-src/verl/utils/kernel/fp8_kernel.py"
    kernel_file.parent.mkdir(parents=True)
    kernel_file.write_text("# kernel\n")
    with pytest.raises(converter.ConversionError, match="visible CUDA device"):
        converter.convert((tmp_path / "bf16").resolve(), tmp_path / "fp8", verl_source=tmp_path / "verl-src")
    assert converter.main(["--source-dir", str(tmp_path / "bf16"), "--target-dir", str(tmp_path / "fp8"), "--verl-source", str(tmp_path / "verl-src")]) == 1
    assert not (tmp_path / "fp8").exists()


def test_gpu_conversion_uses_verls_rollout_kernel_and_records_it(tmp_path, fake_gpu):
    source_tensors = tiny_model(tmp_path / "bf16")
    result = converter.convert((tmp_path / "bf16").resolve(), tmp_path / "fp8", verl_source=fake_gpu["verl_source"])
    quantizer = result["quantizer"]
    assert quantizer["implementation"] == "verl-triton" and quantizer["compute_capability"] == "9.0" and quantizer["device"] == "Fake H200"
    assert quantizer["kernel_sha256"] == converter.sha256(fake_gpu["verl_source"] / "verl/utils/kernel/fp8_kernel.py")
    # The rollout casts every synced weight to bfloat16 and uses 128x128 blocks.
    assert len(fake_gpu["kernel"].calls) == 14 and all(call[0] == torch.bfloat16 and call[2] == [128, 128] for call in fake_gpu["kernel"].calls)
    tensors = read_all(tmp_path / "fp8")
    name = "model.layers.0.mlp.down_proj.weight"
    assert torch.equal(tensors[name + "_scale_inv"], converter.quantize_block_fp8(source_tensors[name])[1] * 2)  # kernel output, squeezed
    marker = json.loads((tmp_path / "fp8" / "fp8_conversion.json").read_text())
    assert marker["quantizer"] == quantizer
    # A later evaluation on another GPU generation reuses the kernel-made copy.
    fake_gpu["capability"] = (8, 9)
    assert converter.convert((tmp_path / "bf16").resolve(), tmp_path / "fp8", verl_source=fake_gpu["verl_source"])["status"] == "reused"


def test_kernel_and_replica_targets_are_never_substituted(tmp_path, fake_gpu):
    tiny_model(tmp_path / "bf16")
    converter.convert((tmp_path / "bf16").resolve(), tmp_path / "replica", kernel="replica")
    with pytest.raises(converter.ConversionError, match="cpu-replica"):
        converter.convert((tmp_path / "bf16").resolve(), tmp_path / "replica", verl_source=fake_gpu["verl_source"])
    converter.convert((tmp_path / "bf16").resolve(), tmp_path / "kernel", verl_source=fake_gpu["verl_source"])
    with pytest.raises(converter.ConversionError, match="verl-triton"):
        converter.convert((tmp_path / "bf16").resolve(), tmp_path / "kernel", kernel="replica")


def test_minimum_capability_and_missing_triton_fail_loudly(tmp_path, fake_gpu):
    tiny_model(tmp_path / "bf16")
    fake_gpu["capability"] = (8, 9)
    with pytest.raises(converter.ConversionError, match="below --min-capability 9.0"):
        converter.convert((tmp_path / "bf16").resolve(), tmp_path / "fp8", verl_source=fake_gpu["verl_source"], min_capability="9.0")
    converter.convert((tmp_path / "bf16").resolve(), tmp_path / "fp8", verl_source=fake_gpu["verl_source"])
    with pytest.raises(converter.ConversionError, match="compute capability 8.9"):
        converter.convert((tmp_path / "bf16").resolve(), tmp_path / "fp8", verl_source=fake_gpu["verl_source"], min_capability="9.0")
    fake_gpu["kernel"] = FakeVerlKernel(triton=False)
    with pytest.raises(converter.ConversionError, match="Triton"):
        converter.convert((tmp_path / "bf16").resolve(), tmp_path / "other", verl_source=fake_gpu["verl_source"])


def test_verification_is_exact_for_kernel_output_on_the_same_generation(tmp_path, fake_gpu):
    tiny_model(tmp_path / "bf16")
    converter.convert((tmp_path / "bf16").resolve(), tmp_path / "fp8", verl_source=fake_gpu["verl_source"])
    report = converter.verify_against_verl(tmp_path / "fp8", (tmp_path / "bf16").resolve(), fake_gpu["verl_source"])
    assert report["implementation"] == "triton" and report["expected_identical"] is True
    assert report["mismatched_elements"] == 0 and report["scale_bitwise_mismatches"] == 0
    fake_gpu["capability"] = (8, 9)
    assert converter.verify_against_verl(tmp_path / "fp8", (tmp_path / "bf16").resolve(), fake_gpu["verl_source"])["expected_identical"] is False
