"""
Recording layer for the simulation.

Captures detailed per-tick state for visualization. The SimulationRecorder
is optionally passed to the Simulation — when present, it records energy
levels, agent decisions, and cumulative metrics every tick.

The output is a JSON-serializable dict that viz.html loads and animates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dc_field, asdict
from typing import Optional


@dataclass
class TickRecord:
    """Everything that happened in one tick, for animation."""
    tick: int
    energy: dict[str, float]              # task_id → energy
    cumulative_extracted: float
    cumulative_waste: float
    cumulative_collisions: int
    agent_actions: list[dict]              # [{agent_id, task_id, success, reward}]


class SimulationRecorder:
    """Records per-tick simulation state for visualization."""

    def __init__(self, task_positions: dict[str, tuple[float, float]],
                 task_types: dict[str, str], couplings: list[dict]):
        self.task_positions = task_positions
        self.task_types = task_types
        self.couplings = couplings
        self.ticks: list[TickRecord] = []

    def record_tick(self, tick: int, field, collisions: int,
                    agent_actions: list[dict]) -> None:
        self.ticks.append(TickRecord(
            tick=tick,
            energy=dict(field._energy),
            cumulative_extracted=field.total_extracted,
            cumulative_waste=field.total_waste,
            cumulative_collisions=collisions,
            agent_actions=agent_actions,
        ))

    def to_dict(self, mechanism: str, final_result) -> dict:
        return {
            "mechanism": mechanism,
            "task_positions": self.task_positions,
            "task_types": self.task_types,
            "couplings": self.couplings,
            "ticks": [
                {
                    "tick": t.tick,
                    "energy": {k: round(v, 2) for k, v in t.energy.items()},
                    "extracted": round(t.cumulative_extracted, 2),
                    "waste": round(t.cumulative_waste, 2),
                    "collisions": t.cumulative_collisions,
                    "actions": t.agent_actions,
                }
                for t in self.ticks
            ],
            "final": {
                "extracted": round(final_result.total_extracted, 2),
                "waste": round(final_result.total_waste, 2),
                "efficiency": round(final_result.efficiency_ratio, 4),
                "tasks_completed": final_result.tasks_completed,
                "collisions": final_result.collisions,
                "gini": round(final_result.gini_coefficient, 4),
            },
        }
