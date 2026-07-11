"""
Synchronicity: Energy-based incentive fields for multi-agent economic coordination.

Core idea: model economic coordination as a physical system. Tasks are nodes
in an energy field. Value chains are edges. Agents read the field, act, and
the field updates according to thermodynamic identities (conservation of
energy, entropy increase). No central planner required.

The physics invariants are enforced as runtime assertions on every field
update. If the math is wrong, the simulation crashes immediately.
"""

from synchronicity.field import EnergyField, FieldSnapshot
from synchronicity.reward_machine import RewardMachine, FieldEvent
from synchronicity.value_chain import ValueChain, TaskType

__version__ = "0.1.0"
__all__ = [
    "EnergyField",
    "FieldSnapshot",
    "RewardMachine",
    "FieldEvent",
    "ValueChain",
    "TaskType",
]
