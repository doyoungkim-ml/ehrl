#!/usr/bin/env python3
"""Idempotently harden AppWorld's ``RequestTracker.save`` against non-JSON
request values (Layer A of the ``...`` / 500 fix).

Background
----------
An under-trained policy sometimes emits code like ``apis.spotify.login(...)``,
passing Python's ``Ellipsis`` (``...``) as an API argument. ``...`` is valid
Python, so the call executes and AppWorld records the request payload verbatim
into ``RequestTracker.requests``. After *every* ``/execute``, AppWorld serializes
the whole request log via FastAPI's ``jsonable_encoder``; that encoder cannot
serialize ``Ellipsis`` and raises, so ``/execute`` returns **HTTP 500**. Because
the poisoned entry stays in the per-task request list, every subsequent
``/execute`` 500s too, and the agent client retries until the trajectory
hard-timeout — wasting wall-clock and forcing reward 0.

This patch coerces any non-JSON-serializable value in the request log to its
``repr`` before serialization. The bad ``...`` call then logs harmlessly, the
API returns its normal "invalid argument" message, ``/execute`` returns 200, and
the bad action becomes ordinary negative RL signal instead of an infra failure.

It edits the *installed* ``appworld`` package (a vendored site-package), so it is
applied by ``setup_server_env_uv.sh`` after each install and can be re-run safely.

Usage (run with the SERVER venv's python):
    "$SERVER_VENV/bin/python" examples/appworld/server/apply_requester_jsonsafe_patch.py

Idempotent: a no-op once the marker is present. Exits non-zero (without writing)
if the expected anchor line is absent, so an AppWorld version bump is noticed
rather than silently skipped.
"""

from __future__ import annotations

import importlib.util
import sys

MARKER = "kailin-jsonsafe-patch"

# Exact source line of RequestTracker.save() in appworld==0.1.3.post1.
ANCHOR = "        requests = [jsonable_encoder(request) for request in self.requests]\n"

REPLACEMENT = (
    "        # kailin-jsonsafe-patch: coerce non-JSON values (e.g. an Ellipsis the\n"
    "        # agent passes as a placeholder API arg) to repr so request-log\n"
    "        # serialization can never 500 /execute. See examples/appworld/server/apply_requester_jsonsafe_patch.py.\n"
    "        def _kailin_json_safe(v):\n"
    "            if isinstance(v, dict):\n"
    "                return {k: _kailin_json_safe(x) for k, x in v.items()}\n"
    "            if isinstance(v, (list, tuple)):\n"
    "                return [_kailin_json_safe(x) for x in v]\n"
    "            try:\n"
    "                import json as _json\n"
    "                _json.dumps(v)\n"
    "                return v\n"
    "            except TypeError:\n"
    "                return repr(v)\n"
    "        requests = [jsonable_encoder(_kailin_json_safe(request)) for request in self.requests]\n"
)


def _requester_path() -> str:
    spec = importlib.util.find_spec("appworld.requester")
    if spec is None or not spec.origin:
        sys.exit("ERROR: appworld.requester not importable in this interpreter.")
    return spec.origin


def main() -> None:
    path = _requester_path()
    with open(path, encoding="utf-8") as fh:
        src = fh.read()

    if MARKER in src:
        print(f"[patch] already applied: {path}")
        return
    if ANCHOR not in src:
        sys.exit(f"ERROR: anchor line not found in {path}.\nThe installed appworld version likely differs from 0.1.3.post1; review RequestTracker.save and update this patch. Patch NOT applied.")

    src = src.replace(ANCHOR, REPLACEMENT, 1)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    print(f"[patch] applied jsonsafe hardening to {path}")


if __name__ == "__main__":
    main()
