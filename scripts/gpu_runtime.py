#!/usr/bin/env python3
"""Discover an interpreter's GPU runtime without importing torch or changing it."""

import argparse
import ast
import importlib.machinery
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


class RuntimeDiscoveryError(RuntimeError):
    """The selected environment or an explicit toolkit cannot be used."""


def package_directory(name, search_path=None):
    """Resolve a top-level package without executing its __init__.py."""
    spec = importlib.util.find_spec(name) if search_path is None else importlib.machinery.PathFinder.find_spec(name, search_path)
    if spec and spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).resolve()
    return None


def torch_cuda_version(torch_directory):
    version_file = torch_directory / "version.py"
    try:
        tree = ast.parse(version_file.read_text())
    except (OSError, SyntaxError) as exc:
        raise RuntimeDiscoveryError(f"Cannot read torch's CUDA version from {version_file}: {exc}") from exc
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(target, ast.Name) and target.id == "cuda" for target in targets):
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str) and re.fullmatch(r"\d+\.\d+", value.value):
                return value.value
    raise RuntimeDiscoveryError(f"{version_file} does not declare a CUDA build; select a CUDA-enabled torch environment.")


def unique_directories(paths):
    result = []
    for path in paths:
        path = Path(path).resolve()
        if path.is_dir() and path not in result:
            result.append(path)
    return result


def merge_library_paths(*values):
    """Keep explicit paths first while deduplicating aliases and omitting empty entries."""
    result = []
    seen = set()
    for value in values:
        for entry in value.split(os.pathsep):
            canonical = str(Path(entry).resolve())
            if entry and canonical not in seen:
                seen.add(canonical)
                result.append(entry)
    return os.pathsep.join(result)


def nvcc_version(toolkit):
    compiler = toolkit / "bin" / "nvcc"
    if not compiler.is_file() or not os.access(compiler, os.X_OK):
        return None
    try:
        result = subprocess.run([str(compiler), "--version"], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"release\s+(\d+\.\d+)", result.stdout + result.stderr)
    return match.group(1) if result.returncode == 0 and match else None


def select_toolkit(cuda_version, candidates, explicit=None, require_compiler=False):
    """Honor explicit selection, otherwise prefer exact CUDA major.minor."""
    warnings = []
    inspected = []
    for candidate in dict.fromkeys(Path(path).expanduser().resolve() for path in candidates):
        version = nvcc_version(candidate)
        inspected.append({"path": str(candidate), "cuda_version": version})
        if explicit is not None:
            if version is None:
                has_runtime = any(path.is_file() for subdir in ("lib", "lib64") for path in (candidate / subdir).glob("libcudart.so*"))
                if has_runtime and not require_compiler:
                    warnings.append(f"Explicit CUDA root {candidate} supplies runtime libraries but no working nvcc; CUDA_HOME will be empty.")
                    continue
                raise RuntimeDiscoveryError(f"Explicit CUDA toolkit {candidate} has no working bin/nvcc. Correct CUDA_HOME/CUDA_PATH or unset it to discover this node's toolkit.")
            if version.split(".")[0] != cuda_version.split(".")[0]:
                raise RuntimeDiscoveryError(f"Explicit CUDA toolkit {candidate} is CUDA {version}, but torch uses CUDA {cuda_version}. Select a toolkit with the same CUDA major.")
    compatible = [item for item in inspected if item["cuda_version"] and item["cuda_version"].split(".")[0] == cuda_version.split(".")[0]]
    selected = next((item for item in compatible if item["cuda_version"] == cuda_version), compatible[0] if compatible else None)
    if selected is None:
        observed = ", ".join(f"{item['path']} (CUDA {item['cuda_version']})" for item in inspected if item["cuda_version"])
        detail = f" Found only incompatible compilers: {observed}." if observed else ""
        message = f"No usable nvcc for torch CUDA {cuda_version}.{detail}"
        if require_compiler:
            raise RuntimeDiscoveryError(message + " Load a matching CUDA toolkit or set CUDA_HOME before builds/JIT that require nvcc.")
        warnings.append(message + " CUDA_HOME is empty; prebuilt runtime mode only. Builds/JIT requiring nvcc need a matching toolkit.")
    elif selected["cuda_version"] != cuda_version:
        warnings.append(f"Using CUDA {selected['cuda_version']} nvcc with torch CUDA {cuda_version} (same major, different minor); extension builds may require an exact toolkit match.")
    return selected, inspected, warnings


def discover_runtime(*, search_path=None, environ=None, conventional_roots=None, require_compiler=False):
    """Inspect selected/inherited sites; overrides keep tests independent of host CUDA."""
    environ = os.environ if environ is None else environ
    torch_directory = package_directory("torch", search_path)
    if torch_directory is None:
        raise RuntimeDiscoveryError(f"No torch package is visible to {sys.executable}; select the training environment's Python.")
    cuda_version = torch_cuda_version(torch_directory)
    vllm_directory = package_directory("vllm", search_path)
    sites = unique_directories(sys.path if search_path is None else search_path)
    # Namespace-package dependencies can be split across a venv and inherited .pth sites.
    # Keep the first provider of each NVIDIA component, following interpreter order.
    nvidia = {}
    for site in sites:
        root = site / "nvidia"
        if root.is_dir():
            for package in sorted(root.iterdir()):
                if package.is_dir():
                    nvidia.setdefault(package.name, package)
    library_candidates = []
    cuda12_home = ""
    cuda_runtime = nvidia.get("cuda_runtime")
    if cuda_runtime:
        for subdir in ("lib64", "lib"):
            directory = cuda_runtime / subdir
            library_candidates.append(directory)
            if any(path.is_file() for path in directory.glob("libcudart.so.12*")):
                cuda12_home = str(cuda_runtime.resolve())
    library_candidates.append(torch_directory / "lib")
    if vllm_directory:
        library_candidates.extend([vllm_directory / "lib", vllm_directory])
    for package in nvidia.values():
        library_candidates.extend([package / "lib64", package / "lib"])
    libraries = unique_directories(library_candidates)

    explicit = environ.get("CUDA_HOME") or environ.get("CUDA_PATH")
    if explicit:
        candidates = [explicit]
    else:
        candidates = [package for name, package in nvidia.items() if name in (f"cu{cuda_version.split('.')[0]}", "cu13", "cuda_nvcc")]
        compiler = shutil.which("nvcc", path=environ.get("PATH", ""))
        if compiler:
            candidates.append(Path(compiler).resolve().parent.parent)
        candidates.extend([Path("/usr/local/cuda"), *sorted(Path("/usr/local").glob("cuda-*"))] if conventional_roots is None else conventional_roots)
    selected, inspected, warnings = select_toolkit(cuda_version, candidates, explicit=explicit, require_compiler=require_compiler)
    cuda_home = selected["path"] if selected else ""
    runtime_override = environ.get("RUNTIME_LD_LIBRARY_PATH") or environ.get("RLLM_RUNTIME_LD_LIBRARY_PATH", "")
    runtime_path = merge_library_paths(runtime_override, os.pathsep.join(map(str, libraries)))
    exports = {
        "CUDA_HOME": cuda_home,
        "CUDA12_RUNTIME_HOME": cuda12_home,
        "RUNTIME_LD_LIBRARY_PATH": runtime_path,
        "RLLM_RUNTIME_LD_LIBRARY_PATH": runtime_path,
        "LD_LIBRARY_PATH": merge_library_paths(runtime_path, environ.get("LD_LIBRARY_PATH", "")),
    }
    if cuda_home:
        exports["PATH"] = os.pathsep.join(value for value in [str(Path(cuda_home) / "bin"), environ.get("PATH", "")] if value)
    return {
        "python": sys.executable,
        "torch_directory": str(torch_directory),
        "torch_cuda_version": cuda_version,
        "vllm_directory": str(vllm_directory) if vllm_directory else None,
        "library_directories": list(map(str, libraries)),
        "compiler": selected,
        "compiler_candidates": inspected,
        "warnings": warnings,
        "exports": exports,
    }


def shell_exports(report):
    return "\n".join(f"export {name}={shlex.quote(value)}" for name, value in report["exports"].items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("shell", "json"), default="json")
    parser.add_argument("--require-compiler", action="store_true", help="Fail unless a same-major CUDA compiler is available")
    args = parser.parse_args()
    try:
        report = discover_runtime(require_compiler=args.require_compiler)
    except RuntimeDiscoveryError as exc:
        print(f"GPU runtime discovery failed: {exc}", file=sys.stderr)
        return 1
    for warning in report["warnings"]:
        print(f"GPU runtime: {warning}", file=sys.stderr)
    print(shell_exports(report) if args.format == "shell" else json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
