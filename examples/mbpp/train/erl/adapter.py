"""MBPP full-solution detection for ERL retry/memory; RAFT is ungated."""


def succeeded(trajectory):
    return bool(trajectory.get("all_passed", False))
