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
import random
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
    """The coordination mechanisms under comparison."""

    GREEDY = "greedy"        # Baseline A: self-interested, no coordination
    PLANNER = "planner"      # Baseline B: central optimal assignment
    FIELD = "field"          # Synchronicity heuristic: energy field navigation
    ACTIVE_INFERENCE = "active_inference"  # FEP-based: minimize expected free energy
    LLM = "llm"              # Real LLM agent (Hermes) reading the field


@dataclass
class SimulationConfig:
    """Configuration for a simulation run.

    Attributes:
        mechanism: Which coordination mechanism to use.
        ticks: Number of simulation steps (epochs).
        injection_schedule: Optional function (tick → list[FieldEvent]) to
            inject capital periodically. None = no ongoing injection.
        seed: Random seed for reproducibility.
        batch_size: Number of agents that commit simultaneously within a tick.
            Within a batch, all agents see the SAME stale field snapshot and
            commit decisions before any execute. This creates real collisions:
            if two agents target the same task, only one succeeds; the other
            wastes the attempt. Between batches, field state propagates.

            batch_size=1 is the original sequential model (implicit coordination).
            batch_size=len(agents) is fully simultaneous (maximum collisions).
            batch_size=2-3 approximates local-network propagation in a tree
            topology: nearby nodes see consistent state, distant nodes see
            stale state.
    """

    mechanism: MechanismType
    ticks: int = 100
    injection_schedule: Optional[Callable[[int], list[FieldEvent]]] = None
    seed: Optional[int] = None
    batch_size: int = 3

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
    collisions: int
    energy_history: list[float]
    entropy_history: list[float]
    extraction_history: list[float]
    end_to_end_throughput: float

    def summary(self) -> str:
        lines = [
            f"  Mechanism:       {self.mechanism.value}",
            f"  Ticks run:       {self.ticks_run}",
            f"  Total extracted: {self.total_extracted:.2f}",
            f"  Total waste:     {self.total_waste:.2f}",
            f"  Efficiency:      {self.efficiency_ratio:.1%}",
            f"  Gini (equity):   {self.gini_coefficient:.3f}",
            f"  Tasks completed: {self.tasks_completed}",
            f"  Collisions:      {self.collisions}",
            f"  End-to-end:      {self.end_to_end_throughput:.1f}",
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

    def __init__(self, chain: ValueChain, field: EnergyField, config: SimulationConfig,
                 recorder=None, use_sigma: bool = False, sigma_tracker=None,
                 angel_alpha: float = 0.0, gradient_gamma: float = 0.0):
        self.chain = chain
        self.field = field
        self.config = config
        self.recorder = recorder
        self.use_angel = angel_alpha > 0 or gradient_gamma > 0

        # Choose reward machine: angel > σ > fixed
        if self.use_angel:
            from synchronicity.angel_reward_machine import AngelRewardMachine
            if sigma_tracker is None:
                from synchronicity.sigma_framework import build_task_complexities, SigmaTracker
                complexities = build_task_complexities(chain)
                sigma_tracker = SigmaTracker(complexities)
            self.sigma_tracker = sigma_tracker
            self.reward_machine = AngelRewardMachine(
                chain, sigma_tracker,
                angel_alpha=angel_alpha,
                gradient_gamma=gradient_gamma,
            )
        elif use_sigma:
            from synchronicity.sigma_reward_machine import SigmaRewardMachine
            if sigma_tracker is None:
                from synchronicity.sigma_framework import build_task_complexities, SigmaTracker
                complexities = build_task_complexities(chain)
                sigma_tracker = SigmaTracker(complexities)
            self.sigma_tracker = sigma_tracker
            self.reward_machine = SigmaRewardMachine(chain, sigma_tracker)
        else:
            self.sigma_tracker = sigma_tracker
            self.reward_machine = RewardMachine(chain)

        self.planner: Optional[PlannerSystem] = None

        if config.mechanism == MechanismType.PLANNER:
            planner_mode = "lookahead" if (use_sigma or self.use_angel) else "energy"
            self.planner = PlannerSystem(
                chain, mode=planner_mode, sigma_tracker=self.sigma_tracker,
            )

    def run(self, agents: list[BaseAgent]) -> SimulationResult:
        """Run the simulation using batch-based commitment.

        Each tick (epoch) is split into batches of `batch_size` agents.
        Within a batch:
          1. All agents in the batch observe the SAME field snapshot
          2. All agents commit decisions based on that snapshot
          3. Decisions execute in random order — collisions on the same task
             cause waste (second agent fails)
        Between batches, the field state propagates.

        This models a tree topology: agents in the same batch are "local"
        to each other and see consistent state. Agents in different batches
        see propagated state from earlier batches.

        Why this matters: with batch_size >= 2, greedy agents collide on
        high-energy tasks. Field-aware agents (who consider the full value
        chain) spread across tasks because their scoring includes downstream
        depth, not just raw energy.
        """
        field = self.field
        machine = self.reward_machine
        batch_size = self.config.batch_size
        random.shuffle(agents)  # randomize batch membership each tick

        energy_hist: list[float] = []
        entropy_hist: list[float] = []
        extraction_hist: list[float] = []
        tasks_completed = 0
        tick_collisions = 0

        for tick in range(self.config.ticks):
            tick_actions: list[dict] = []
            tick_collisions = 0

            # Periodic capital injection (keeps the economy running)
            if self.config.injection_schedule:
                for event in self.config.injection_schedule(tick):
                    machine.apply(field, event)

            # Split agents into batches
            for batch_start in range(0, len(agents), batch_size):
                batch = agents[batch_start : batch_start + batch_size]

                # Phase 1: all agents in batch observe the SAME snapshot
                snapshot = field.snapshot()
                for agent in batch:
                    agent.observe(snapshot)

                # Planner assigns within this batch
                if self.config.mechanism == MechanismType.PLANNER and self.planner:
                    self.planner.assign(
                        [a for a in batch if isinstance(a, PlannerAgent)],
                        snapshot,
                    )

                # Phase 2: all agents commit decisions
                decisions = [(agent, agent.decide()) for agent in batch]

                # Phase 3: execute in random order (simulates network race)
                random.shuffle(decisions)

                for agent, decision in decisions:
                    if decision.task_id is None:
                        continue
                    if field.energy_at(decision.task_id) < 0.5:
                        # Task already drained by another agent in this batch
                        agent.receive_reward(0.0, success=False)
                        tick_collisions += 1
                        tick_actions.append({
                            "agent_id": agent.id,
                            "task_id": decision.task_id,
                            "success": False,
                            "reward": 0.0,
                            "collision": True,
                        })
                        continue

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
                        tick_actions.append({
                            "agent_id": agent.id,
                            "task_id": task_id,
                            "success": True,
                            "reward": round(reward, 2),
                            "collision": False,
                        })
                    else:
                        event = FieldEvent(
                            event_type=EventType.TASK_FAILED,
                            task_id=task_id,
                            agent_id=agent.id,
                        )
                        machine.apply(field, event)
                        agent.receive_reward(0.0, success=False)
                        tick_collisions += 1
                        tick_actions.append({
                            "agent_id": agent.id,
                            "task_id": task_id,
                            "success": False,
                            "reward": 0.0,
                            "collision": False,
                        })

            # Apply angel coefficient tick decay (gradient fading)
            if hasattr(self.reward_machine, 'apply_tick_decay'):
                self.reward_machine.apply_tick_decay(field)

            # Verify conservation (angel machine has its own check)
            if hasattr(self.reward_machine, 'check_conservation'):
                self.reward_machine.check_conservation(field)
            else:
                field.check_conservation()

            # Record metrics
            energy_hist.append(field.total_energy())
            entropy_hist.append(field.total_waste)
            extraction_hist.append(field.total_extracted)

            # Record for visualization
            if self.recorder:
                self.recorder.record_tick(
                    tick, field, sum(a.tasks_failed for a in agents),
                    tick_actions,
                )

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
        total_collisions = sum(a.tasks_failed for a in agents)

        # End-to-end throughput: how much energy reached TERMINAL tasks
        # (tasks with no downstream couplings — the final market/retail nodes).
        # This measures whether the value chain actually flowed end-to-end.
        terminal_ids = [
            tid for tid in self.chain.all_task_ids()
            if not self.chain.downstream(tid)
        ]
        end_to_end = sum(
            field.history[-1].get(tid, 0.0)
            for tid in terminal_ids
            if field.history
        )

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
            collisions=total_collisions,
            energy_history=energy_hist,
            entropy_history=entropy_hist,
            extraction_history=extraction_hist,
            end_to_end_throughput=end_to_end,
        )
