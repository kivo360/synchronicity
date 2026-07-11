"""
Simulation harness: runs experiments comparing coordination mechanisms.

This is the testbed for the core claim of Synchronicity:

    An energy-based incentive field causes a swarm of self-interested agents
    to self-organize into a more efficient configuration than pure greedy
    optimization, approaching the efficiency of a central planner, while
    maintaining local autonomy and adaptability.

Usage:
    from synchronicity.simulation import Simulation, SimulationConfig, MechanismType
    from synchronicity.scenarios import build_trash_cleanup_scenario

    chain, field, agents = build_trash_cleanup_scenario()
    sim = Simulation(chain, field, SimulationConfig(mechanism=MechanismType.FIELD))
    results = sim.run(agents, ticks=100)

The simulation runs identically across all three mechanisms — only the
agent decision-making differs. This isolates the variable being tested.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Optional, Callable

from synchronicity.field import EnergyField, FieldSnapshot
from synchronicity.reward_machine import RewardMachine, FieldEvent, EventType
from synchronicity.value_chain import ValueChain
from synchronicity.agents import (
    BaseAgent,
    GreedyAgent,
    FieldAgent,
    PlannerAgent,
    PlannerSystem,
)

logger = logging.getLogger(__name__)


class MechanismType(str, Enum):
    """The three coordination mechanisms under comparison."""

    GREEDY = "greedy"        # Baseline A: self-interested, no coordination
    PLANNER = "planner"      # Baseline B: central optimal assignment
    FIELD = "field"          # Synchronicity: energy field navigation


@dataclass
class SimulationConfig:
    """Configuration for a simulation run.

    Attributes:
        mechanism: Which coordination mechanism to use.
        ticks: Number of simulation steps.
        injection_schedule: Optional function (tick → list[FieldEvent]) to
            inject capital periodically. None = no ongoing injection.
        seed: Random seed for reproducibility.
    """

    mechanism: MechanismType
    ticks: int = 100
    injection_schedule: Optional[Callable[[int], list[FieldEvent]]] = None
    seed: Optional[int] = None

    def __post_init__(self):
        if self.seed is not None:
            import random
            random.seed(self.seed)


@dataclass
class SimulationResult:
    """Results from a completed simulation run.

    These are the metrics that matter for comparing mechanisms:

    - total_extracted: Total realized value (profit) the system generated.
        Higher = more economically productive.
    - total_waste: Total entropy (wasted energy). Lower = more efficient.
    - efficiency_ratio: extracted / (extracted + waste). Higher = better.
    - agent_rewards: Per-agent reward distribution. Equality of distribution
        matters — a mechanism that concentrates all reward in one agent
        isn't good even if total is high.
    - gini_coefficient: Measures reward inequality (0 = perfect equality,
        1 = one agent gets everything). Lower = more equitable.
    - tasks_completed: Total successful task completions.
    - energy_history: Per-tick total energy in the field (for plotting).
    - entropy_history: Per-tick cumulative entropy (for plotting).
    - extraction_history: Per-tick cumulative extracted value.
    """

    mechanism: MechanismType
    ticks_run: int
    total_extracted: float
    total_waste: float
    efficiency_ratio: float
    agent_rewards: dict[str, float]
    agent_stats: list[dict]
    gini_coefficient: float
    tasks_completed: int
    energy_history: list[float]
    entropy_history: list[float]
    extraction_history: list[float]

    def summary(self) -> str:
        lines = [
            f"  Mechanism:       {self.mechanism.value}",
            f"  Ticks run:       {self.ticks_run}",
            f"  Total extracted: {self.total_extracted:.2f}",
            f"  Total waste:     {self.total_waste:.2f}",
            f"  Efficiency:      {self.efficiency_ratio:.1%}",
            f"  Gini (equity):   {self.gini_coefficient:.3f}",
            f"  Tasks completed: {self.tasks_completed}",
            f"  Agents:          {len(self.agent_rewards)}",
        ]
        return "\n".join(lines)


def gini_coefficient(values: list[float]) -> float:
    """Calculate the Gini coefficient (inequality measure).

    0 = perfect equality, 1 = one entity has everything.
    """
    if not values or sum(values) == 0:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    cumsum = sum((i + 1) * v for i, v in enumerate(sorted_vals))
    total = sum(sorted_vals)
    return (2 * cumsum) / (n * total) - (n + 1) / n


class Simulation:
    """The experiment runner.

    Manages the simulation loop: agents observe → decide → execute →
    field updates. Runs identically for all three mechanisms.

    For PLANNER mechanism, the simulation instantiates a PlannerSystem
    that assigns agents optimally each tick. For GREEDY and FIELD,
    agents decide independently.
    """

    def __init__(self, chain: ValueChain, field: EnergyField, config: SimulationConfig):
        self.chain = chain
        self.field = field
        self.config = config
        self.reward_machine = RewardMachine(chain)
        self.planner: Optional[PlannerSystem] = None

        if config.mechanism == MechanismType.PLANNER:
            self.planner = PlannerSystem(chain)

    def run(self, agents: list[BaseAgent]) -> SimulationResult:
        """Run the simulation for the configured number of ticks."""
        field = self.field
        machine = self.reward_machine

        energy_hist: list[float] = []
        entropy_hist: list[float] = []
        extraction_hist: list[float] = []
        tasks_completed = 0

        for tick in range(self.config.ticks):
            # Periodic capital injection (keeps the economy running)
            if self.config.injection_schedule:
                for event in self.config.injection_schedule(tick):
                    machine.apply(field, event)

            snapshot = field.snapshot()

            # Decision phase — collect all decisions first
            if self.config.mechanism == MechanismType.PLANNER:
                self.planner.assign(
                    [a for a in agents if isinstance(a, PlannerAgent)],
                    snapshot,
                )

            # Execution phase — process agents sequentially, re-reading the
            # field between actions so later agents see updated energy levels.
            # This simulates real-time field updates: when agent A drains a
            # task, agent B sees the updated incentive and picks something else.
            for agent in agents:
                # Re-snapshot so the agent sees the current field state
                agent.observe(field.snapshot())
                decision = agent.decide()

                if decision.task_id is None:
                    continue

                # Check if there's still energy at the task
                if field.energy_at(decision.task_id) < 0.5:
                    continue  # task exhausted, skip

                efficiency, task_id = agent.execute(decision)

                if efficiency > 0:
                    event = FieldEvent(
                        event_type=EventType.TASK_COMPLETED,
                        task_id=task_id,
                        agent_id=agent.id,
                        efficiency=efficiency,
                    )
                    reward = machine.apply(field, event)
                    agent.receive_reward(reward, success=True)
                    tasks_completed += 1
                else:
                    event = FieldEvent(
                        event_type=EventType.TASK_FAILED,
                        task_id=task_id,
                        agent_id=agent.id,
                    )
                    machine.apply(field, event)
                    agent.receive_reward(0.0, success=False)

            # Record metrics
            energy_hist.append(field.total_energy())
            entropy_hist.append(field.total_waste)
            extraction_hist.append(field.total_extracted)

            # Early termination only if truly dead — no energy AND no injection
            if (
                field.total_energy() < 0.01
                and self.config.injection_schedule is None
            ):
                break

        # Compute final metrics
        rewards = [a.total_reward for a in agents]
        total_extracted = field.total_extracted
        total_waste = field.total_waste
        efficiency = total_extracted / max(total_extracted + total_waste, 0.001)

        return SimulationResult(
            mechanism=self.config.mechanism,
            ticks_run=tick + 1,
            total_extracted=total_extracted,
            total_waste=total_waste,
            efficiency_ratio=efficiency,
            agent_rewards={a.id: a.total_reward for a in agents},
            agent_stats=[a.stats() for a in agents],
            gini_coefficient=gini_coefficient(rewards),
            tasks_completed=tasks_completed,
            energy_history=energy_hist,
            entropy_history=entropy_hist,
            extraction_history=extraction_hist,
        )
