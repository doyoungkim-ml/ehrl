"""Mastermind full-solution detection for ERL retry/memory; RAFT is ungated."""


def succeeded(trajectory):
    # Mastermind max-reward aggregation is 1 only for a complete correct code.
    return float(trajectory["trajectory_reward"]) >= 1.0
