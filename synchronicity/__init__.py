"""
Synchronicity: Energy-based incentive fields for multi-agent economic coordination.

Core idea: model economic coordination as a physical system. Tasks are nodes
in an energy field. Value chains are edges. Agents read the field, act, and
the field updates according to thermodynamic identities (conservation of
energy, entropy increase). No central planner required.

The physics invariants are enforced as runtime assertions on every field
update. If the math is wrong, the simulation crashes immediately.

Five agent mechanisms:
  1. Greedy           — softmax over raw energy (baseline)
  2. Planner          — central optimal assignment (upper bound)
  3. Field            — heuristic field scoring with learned competition
  4. Active Inference  — Free Energy Principle (FEP) based scoring
  5. LLM              — real LLM agent (Hermes) reasoning about the field

Value chain topologies can be:
  - Hand-coded (scenarios.py: trash-cleanup, crossroads, cascade, burst)
  - Auto-discovered from business event logs (process_ingestion.py via PM4Py)
"""

from synchronicity.field import EnergyField, FieldSnapshot
from synchronicity.reward_machine import RewardMachine, FieldEvent, EventType
from synchronicity.value_chain import ValueChain, TaskType, Task, Coupling
from synchronicity.agents import (
    AgentCapability, AgentDecision, BaseAgent,
    GreedyAgent, FieldAgent, PlannerAgent, PlannerSystem,
)
from synchronicity.active_inference import ActiveInferenceAgent

__version__ = "0.1.0"
__all__ = [
    "EnergyField",
    "FieldSnapshot",
    "RewardMachine",
    "FieldEvent",
    "EventType",
    "ValueChain",
    "TaskType",
    "Task",
    "Coupling",
    "AgentCapability",
    "AgentDecision",
    "BaseAgent",
    "GreedyAgent",
    "FieldAgent",
    "PlannerAgent",
    "PlannerSystem",
    "ActiveInferenceAgent",
]
