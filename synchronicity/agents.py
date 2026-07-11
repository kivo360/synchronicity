"""
Agent protocol and baseline implementations.

This module defines how agents interact with the energy field. An agent
reads a field snapshot, decides which task to attempt, executes it, and
reports an outcome. The field then updates.

Three agent types are provided:

  1. GreedyAgent — always picks the task with highest current energy.
     No strategic reasoning. Baseline A in the experiment design.
     Represents the "invisible hand" — pure self-interested local optimization.

  2. PlannerAgent — a central planner with full knowledge of the value chain.
     Assigns agents to tasks using a globally optimal strategy (solves an
     assignment problem to maximize total energy throughput). Baseline B.
     Represents the "centralized control" extreme.

  3. FieldAgent — reads the energy field structurally, considering not just
     raw energy but downstream value chain connections, capability matching,
     and efficiency estimates. This is the Synchronicity mechanism — agents
     navigate the incentive landscape intelligently, without a central planner.

The protocol is designed to be implementable by an LLM agent (Hermes, raw
OpenAI, etc.). The FieldAgent here is a rule-based approximation of what
an LLM would do — it demonstrates the mechanism works before adding LLM
reasoning on top.
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dc_field
from typing import Optional

import numpy as np

from synchronicity.field import EnergyField, FieldSnapshot
from synchronicity.value_chain import ValueChain, Task, TaskType

logger = logging.getLogger(__name__)


@dataclass
class AgentCapability:
    """An agent's capability profile.

    Attributes:
        id: Unique agent identifier.
        capabilities: Set of capability tags this agent can perform.
            Tasks requiring capabilities the agent lacks are either
            penalized (reduced efficiency) or impossible.
        base_efficiency: How well the agent performs tasks it CAN do.
            1.0 = perfect, 0.5 = mediocre.
        learning_rate: Optional rate at which efficiency improves with
            experience on a task type.
    """

    id: str
    capabilities: frozenset[str] = frozenset()
    base_efficiency: float = 1.0
    learning_rate: float = 0.0


@dataclass
class AgentDecision:
    """An agent's decision about what to do this turn.

    Attributes:
        task_id: The task the agent chose to attempt (None = idle/decline).
        expected_efficiency: Agent's estimate of how well it'll do.
        reasoning: Human-readable explanation (important for LLM agents
            and for debugging).
    """

    task_id: Optional[str]
    expected_efficiency: float = 1.0
    reasoning: str = ""


class BaseAgent(ABC):
    """Protocol for all agent types.

    An agent's lifecycle in the simulation loop:
        1. observe(field_snapshot) — see the current state
        2. decide() — choose a task
        3. execute() — attempt the task (simulated)
        4. Field updates via RewardMachine
        5. Receive reward feedback
    """

    def __init__(self, capability: AgentCapability):
        self.capability = capability
        self.total_reward: float = 0.0
        self.tasks_completed: int = 0
        self.tasks_failed: int = 0
        self._last_snapshot: Optional[FieldSnapshot] = None

    @property
    def id(self) -> str:
        return self.capability.id

    def observe(self, snapshot: FieldSnapshot) -> None:
        """Receive the current field state."""
        self._last_snapshot = snapshot

    @abstractmethod
    def decide(self) -> AgentDecision:
        """Choose which task to attempt this turn."""
        ...

    def execute(self, decision: AgentDecision) -> tuple[float, str]:
        """Attempt the chosen task. Returns (efficiency, task_id).

        In the simulation, task execution is stochastic — efficiency varies.
        In a real LLM agent, this is where the agent actually does work
        (makes tool calls, runs code, etc.) and reports outcomes.
        """
        if decision.task_id is None:
            return 0.0, ""

        # Check capability match
        task_caps = self._last_snapshot.task_capabilities.get(
            decision.task_id, frozenset()
        )
        if task_caps and not task_caps.intersection(self.capability.capabilities):
            # Agent can't do this task at all
            return 0.0, decision.task_id

        # Add some noise to simulate real-world variance
        base = self.capability.base_efficiency
        noise = random.gauss(0, 0.05)
        efficiency = max(0.0, min(1.0, base + noise))

        if efficiency > 0.3:
            return efficiency, decision.task_id
        else:
            return 0.0, decision.task_id  # failed

    def receive_reward(self, reward: float, success: bool) -> None:
        """Called after the field processes the agent's action."""
        self.total_reward += reward
        if success:
            self.tasks_completed += 1
        else:
            self.tasks_failed += 1

    def stats(self) -> dict:
        return {
            "id": self.id,
            "reward": round(self.total_reward, 2),
            "completed": self.tasks_completed,
            "failed": self.tasks_failed,
            "efficiency": round(
                self.tasks_completed / max(self.tasks_completed + self.tasks_failed, 1), 3
            ),
        }


class GreedyAgent(BaseAgent):
    """Baseline A: pure self-interest, no strategic reasoning.

    Always picks the task with the highest current energy allocation.
    No consideration of value chains, downstream effects, or capability fit.

    Represents the "invisible hand" — each agent optimizing locally.
    The theory predicts this leads to clumping and inefficiency.
    """

    def decide(self) -> AgentDecision:
        snap = self._last_snapshot
        if snap is None:
            return AgentDecision(task_id=None, reasoning="no snapshot")

        # Pick highest-energy task the agent is capable of doing
        best_task = None
        best_energy = 0.0
        for task_id, energy in snap.task_energy.items():
            if energy <= 0:
                continue
            caps = snap.task_capabilities.get(task_id, frozenset())
            if caps and not caps.intersection(self.capability.capabilities):
                continue  # can't do it
            if energy > best_energy:
                best_energy = energy
                best_task = task_id

        if best_task is None:
            return AgentDecision(task_id=None, reasoning="no available tasks")

        return AgentDecision(
            task_id=best_task,
            expected_efficiency=self.capability.base_efficiency,
            reasoning=f"greedy: highest energy task ({best_energy:.1f})",
        )


class PlannerAgent(BaseAgent):
    """Baseline B: central planner with global knowledge.

    This agent is part of a coordinated system where one planner assigns
    all agents optimally. The PlannerAgent class itself just receives
    assignments — the PlannerSystem class does the optimization.

    Represents the "centralized control" extreme — efficient but requires
    full knowledge, single point of failure, no local autonomy.
    """

    def __init__(self, capability: AgentCapability):
        super().__init__(capability)
        self._assignment: Optional[str] = None

    def set_assignment(self, task_id: Optional[str]) -> None:
        """Called by the PlannerSystem to assign this agent a task."""
        self._assignment = task_id

    def decide(self) -> AgentDecision:
        if self._assignment is None:
            return AgentDecision(task_id=None, reasoning="no assignment")

        task_id = self._assignment
        self._assignment = None  # consume the assignment

        return AgentDecision(
            task_id=task_id,
            expected_efficiency=self.capability.base_efficiency,
            reasoning=f"planner-assigned",
        )


class PlannerSystem:
    """Central planner that assigns agents to tasks optimally.

    Uses the Hungarian algorithm to solve the assignment problem:
    maximize total energy processed, given agent capabilities and task
    energy levels. Each tick, it reassigns all agents.

    This is the strongest possible baseline — it has perfect information
    and solves the optimization problem exactly. If the Synchronicity
    mechanism (FieldAgent) gets close to this, that's a strong result.
    """

    def __init__(self, chain: ValueChain):
        self.chain = chain

    def assign(
        self,
        agents: list[PlannerAgent],
        snapshot: FieldSnapshot,
    ) -> dict[str, Optional[str]]:
        """Assign each agent to a task for this tick.

        Returns: agent_id → task_id (or None if unassigned).
        """
        from scipy.optimize import linear_sum_assignment

        task_ids = [tid for tid, e in snapshot.task_energy.items() if e > 0.1]
        if not task_ids:
            return {a.id: None for a in agents}

        n_agents = len(agents)
        n_tasks = len(task_ids)

        # Build cost matrix (negative because we maximize)
        # Rows = agents, Cols = tasks
        # Value = energy the agent would process if assigned to this task
        cost_matrix = np.zeros((n_agents, n_tasks))
        for i, agent in enumerate(agents):
            for j, task_id in enumerate(task_ids):
                caps = snapshot.task_capabilities.get(task_id, frozenset())
                energy = snapshot.task_energy[task_id]

                if caps and not caps.intersection(agent.capability.capabilities):
                    cost_matrix[i][j] = 0  # can't do it
                else:
                    cost_matrix[i][j] = energy * agent.capability.base_efficiency

        # Solve (minimize negative = maximize positive)
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)

        assignments: dict[str, Optional[str]] = {a.id: None for a in agents}
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r][c] > 0:
                agent = agents[r]
                task_id = task_ids[c]
                assignments[agent.id] = task_id
                agent.set_assignment(task_id)

        return assignments


class FieldAgent(BaseAgent):
    """The Synchronicity mechanism: agents that read the energy field
    structurally and reason about value chains.

    Unlike the greedy agent (which only sees raw energy), the FieldAgent
    considers:

    1. Raw energy at the task (immediate incentive)
    2. Downstream value chain depth (how much the task matters to the system)
    3. Capability match (how well the agent can do it)
    4. Competition (how many other agents are likely to pick the same task)

    This is the core claim: agents navigating the incentive field intelligently
    achieve near-planner efficiency without centralized control.

    The scoring function is intentionally transparent and tunable. In a
    full LLM implementation, the agent would reason about these same factors
    in natural language instead of through a formula.
    """

    def __init__(
        self,
        capability: AgentCapability,
        chain: ValueChain,
        downstream_weight: float = 0.3,
        capability_weight: float = 0.2,
        competition_penalty: float = 0.1,
    ):
        super().__init__(capability)
        self.chain = chain
        self.downstream_weight = downstream_weight
        self.capability_weight = capability_weight
        self.competition_penalty = competition_penalty

    def decide(self) -> AgentDecision:
        snap = self._last_snapshot
        if snap is None:
            return AgentDecision(task_id=None, reasoning="no snapshot")

        scores: dict[str, float] = {}
        reasoning_parts: dict[str, str] = {}

        for task_id, energy in snap.task_energy.items():
            if energy <= 0.1:
                continue

            # Filter: skip tasks the agent cannot do
            task_caps = snap.task_capabilities.get(task_id, frozenset())
            if task_caps and not task_caps.intersection(self.capability.capabilities):
                continue

            # Factor 1: raw energy (immediate incentive)
            score = energy

            # Factor 2: downstream value chain depth
            downstream = snap.downstream.get(task_id, [])
            chain_depth = len(snap.downstream.get(task_id, []))
            chain_bonus = chain_depth * self.downstream_weight * 10
            score += chain_bonus

            # Factor 3: capability match (all tasks that pass the filter
            # have at least partial match; exact match gets a bonus)
            if task_caps:
                overlap = len(task_caps.intersection(self.capability.capabilities))
                match = overlap / len(task_caps)
                cap_bonus = match * self.capability_weight * 50
                score += cap_bonus
            else:
                cap_bonus = 0

            scores[task_id] = score
            reasoning_parts[task_id] = (
                f"E={energy:.1f} + chain_depth={chain_depth}"
                f" + cap_match={cap_bonus:.1f}"
                f" = {score:.1f}"
            )

        if not scores:
            return AgentDecision(task_id=None, reasoning="no viable tasks")

        # Pick the best-scored task
        best_task = max(scores, key=scores.get)

        return AgentDecision(
            task_id=best_task,
            expected_efficiency=self.capability.base_efficiency,
            reasoning=f"field: {reasoning_parts[best_task]}",
        )
