"""
Prompt Adapter Module for ACE Reflector

This module provides environment-specific adapters for formatting trajectory data
into Reflector prompt template variables.

Design Pattern:
- ReflectorAdapter: Protocol defining the adapter interface
- DefaultReflectorAdapter: Backward-compatible adapter (legacy variables)
- Environment-specific adapters: Located in example directories (e.g., mbpp_adapter.py)

Usage:
    # Default adapter (backward compatible with FrozenLake, AppWorld)
    from rllm.algorithms.ace.context import DefaultReflectorAdapter
    adapter = DefaultReflectorAdapter()

    # Environment-specific adapter (e.g., for MBPP)
    from mbpp_adapter import MBPPReflectorAdapter
    adapter = MBPPReflectorAdapter(max_turns=5)

    # Pass to ACEContextRule
    rule = ACEContextRule(config, rollout_engine=engine, reflector_adapter=adapter)

Template Variables:
    DefaultReflectorAdapter provides:
        - {{outcome}}: "SUCCESS" or "FAILED"
        - {{failure_info}}: Human-readable failure explanation
        - {{conversation_history}}: Formatted episode history
        - {{playbook}}: Current playbook content
        - {{trajectory_summary}}: Structured summary (also available)

    MBPPReflectorAdapter provides:
        - {{trajectory_summary}}: Rich summary with turns, reward, termination
        - {{conversation_history}}: Formatted episode history
        - {{playbook}}: Current playbook content
        (Does NOT provide {{outcome}} or {{failure_info}} - uses {{trajectory_summary}} instead)
"""

from .base import DefaultReflectorAdapter, ReflectorAdapter

__all__ = [
    "ReflectorAdapter",
    "DefaultReflectorAdapter",
]
