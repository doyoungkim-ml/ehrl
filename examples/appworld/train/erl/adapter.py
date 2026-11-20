"""AppWorld full-solution detection for ERL retry/memory; RAFT is ungated."""


def succeeded(trajectory):
    return float(trajectory["trajectory_reward"]) >= 1.0
