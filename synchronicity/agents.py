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


def _softmax_sample(scores: dict[str, float], temperature: float = 1.0) -> str:
    """Sample a key proportional to exp(score / temperature).

    This is the standard Boltzmann/softmax policy from reinforcement learning.
    Used by both GreedyAgent and FieldAgent so the selection MECHANISM is
    identical — only the SCORES differ (raw energy vs. structured field score).
    """
    if not scores:
        raise ValueError("Cannot softmax-sample from empty scores")
    if len(scores) == 1:
        return next(iter(scores))

    # Subtract max for numerical stability
    max_score = max(scores.values())
    exp_scores = {
        k: pow(2.718281828, (v - max_score) / temperature)
        for k, v in scores.items()
    }
    total = sum(exp_scores.values())

    r = random.random() * total
    cumulative = 0.0
    for key, exp_s in exp_scores.items():
        cumulative += exp_s
        if r <= cumulative:
            return key
    return next(iter(scores))  # fallback


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

    Samples tasks proportional to raw energy allocation using softmax.
    No consideration of value chains, downstream effects, or capability fit.

    Represents the "invisible hand" — each agent optimizing locally.
    The theory predicts this leads to clumping and inefficiency.
    """

    def decide(self) -> AgentDecision:
        snap = self._last_snapshot
        if snap is None:
            return AgentDecision(task_id=None, reasoning="no snapshot")

        # Score = raw energy only. No structure, no competition awareness.
        scores: dict[str, float] = {}
        for task_id, energy in snap.task_energy.items():
            if energy <= 0.1:
                continue
            caps = snap.task_capabilities.get(task_id, frozenset())
            if caps and not caps.intersection(self.capability.capabilities):
                continue
            scores[task_id] = energy

        if not scores:
            return AgentDecision(task_id=None, reasoning="no available tasks")

        # Softmax sampling over raw energy (same selection mechanism as
        # FieldAgent, but over UNSTRUCTURED scores)
        best_task = _softmax_sample(scores, temperature=1.0)

        return AgentDecision(
            task_id=best_task,
            expected_efficiency=self.capability.base_efficiency,
            reasoning=f"greedy: raw energy softmax ({scores[best_task]:.1f})",
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
    maximize total value processed, given agent capabilities and task
    energy levels. Each tick, it reassigns all agents.

    This is the strongest possible baseline — it has perfect information
    and solves the optimization problem exactly. If the Synchronicity
    mechanism (FieldAgent) gets close to this, that's a strong result.

    Supports two scoring modes:
      - 'energy': cost matrix = energy × base_efficiency (original, myopic)
      - 'sigma':  cost matrix = energy × σ_chain (σ-aware, uses composition rule)
    """

    def __init__(self, chain: ValueChain, mode: str = "energy", sigma_tracker=None):
        self.chain = chain
        self.mode = mode
        self.sigma_tracker = sigma_tracker

    def _score_assignment(self, agent, task_id: str, energy: float,
                          snapshot: FieldSnapshot) -> float:
        """Score the value of assigning an agent to a task.

        In 'energy' mode: score = energy × base_efficiency (myopic)
        In 'sigma' mode:  score = energy × σ_chain (composition rule)
        In 'lookahead' mode: score = σ_chain × (energy + expected_downstream_energy)
        """
        if self.mode == "lookahead" and self.sigma_tracker:
            # Multi-horizon: score includes expected downstream energy
            # that will be unlocked by completing this task.
            # This is a 2-tick lookahead: what energy arrives downstream
            # if I complete this task now?
            sigma_chain = self.sigma_tracker.chain_sigma(
                agent.id, task_id, self.chain, snapshot, max_depth=3,
            )

            # Estimate downstream energy that will be unlocked
            downstream_energy = 0.0
            for ds_id, coeff in snapshot.downstream.get(task_id, []):
                ds_energy = snapshot.task_energy.get(ds_id, 0.0)
                # Energy arriving downstream = propagated fraction of current task energy
                downstream_energy += energy * coeff * sigma_chain * 0.3  # propagation factor

            return (energy + downstream_energy) * sigma_chain

        elif self.mode == "sigma" and self.sigma_tracker:
            sigma_chain = self.sigma_tracker.chain_sigma(
                agent.id, task_id, self.chain, snapshot, max_depth=3,
            )
            return energy * sigma_chain

        else:
            return energy * agent.capability.base_efficiency

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
        cost_matrix = np.zeros((n_agents, n_tasks))
        for i, agent in enumerate(agents):
            for j, task_id in enumerate(task_ids):
                caps = snapshot.task_capabilities.get(task_id, frozenset())
                energy = snapshot.task_energy[task_id]

                if caps and not caps.intersection(agent.capability.capabilities):
                    cost_matrix[i][j] = 0  # can't do it
                else:
                    cost_matrix[i][j] = self._score_assignment(
                        agent, task_id, energy, snapshot,
                    )

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
    uses three strategic signals that greedy ignores:

    1. Learned competition model: The agent tracks energy depletion patterns
       across ticks to learn WHERE other agents are likely to go. If task X
       was drained last tick (by another agent), the agent knows it has
       competition there. This is an empirical model built from observation,
       not a heuristic.

    2. Downstream multiplier: Completing a bridge task unlocks energy
       downstream. The true value of sorting isn't just its own energy —
       it's its energy PLUS the downstream energy it propagates to (weighted
       by coupling coefficients).

    3. Capability exclusivity: If the agent is one of few who can do a
       bottleneck task, the system needs them there. The field agent values
       tasks higher when fewer competitors can do them.

    The scoring function is intentionally transparent and tunable. In a
    full LLM implementation, the agent would reason about these same factors
    in natural language instead of through a formula.
    """

    def __init__(
        self,
        capability: AgentCapability,
        chain: ValueChain,
        downstream_weight: float = 0.5,
        competition_weight: float = 1.0,
        exclusivity_weight: float = 0.3,
        learning_rate: float = 0.15,
    ):
        super().__init__(capability)
        self.chain = chain
        self.downstream_weight = downstream_weight
        self.competition_weight = competition_weight
        self.exclusivity_weight = exclusivity_weight
        self.learning_rate = learning_rate

        # Learned competition model: task_id → estimated probability of
        # being targeted by another agent in the next tick.
        # Initialized to uniform (no prior knowledge).
        self._competition_pressure: dict[str, float] = {}

        # Track energy levels observed last tick to detect depletion
        self._last_energy_observed: dict[str, float] = {}

    def observe(self, snapshot: FieldSnapshot) -> None:
        """Update the learned competition model from observed state changes.

        Between the previous snapshot and the current one, energy at some
        tasks dropped. Those drops indicate other agents completed (or
        attempted) those tasks. We use this signal to learn where competitors
        are concentrating.
        """
        # On first observation, just store baseline
        if not self._last_energy_observed:
            self._last_energy_observed = dict(snapshot.task_energy)
            self._last_snapshot = snapshot
            return

        # Detect tasks where energy dropped significantly (indicating another
        # agent completed the task and drained it)
        for task_id, current_energy in snapshot.task_energy.items():
            prev_energy = self._last_energy_observed.get(task_id, 0.0)
            drop = prev_energy - current_energy

            if drop > 5.0:
                # Another agent (or agents) hit this task. Update pressure
                # estimate using exponential moving average.
                old_pressure = self._competition_pressure.get(task_id, 0.0)
                # Magnitude of the pressure update proportional to energy drop
                # relative to total energy in the system
                total_e = max(sum(snapshot.task_energy.values()), 1.0)
                intensity = min(drop / total_e * 10, 1.0)
                new_pressure = old_pressure + self.learning_rate * (intensity - old_pressure)
                self._competition_pressure[task_id] = new_pressure
            elif drop <= 0:
                # Energy didn't drop → no competition here → decay pressure
                old_pressure = self._competition_pressure.get(task_id, 0.0)
                self._competition_pressure[task_id] = old_pressure * (1 - self.learning_rate * 0.5)

        # Update baseline
        self._last_energy_observed = dict(snapshot.task_energy)
        self._last_snapshot = snapshot

    def _downstream_energy_potential(
        self, task_id: str, snapshot: FieldSnapshot
    ) -> float:
        """Calculate total downstream energy unlocked by completing this task.

        This is the bridge multiplier: completing sorting doesn't just give
        you sorting's energy — it propagates energy to compost and recycling,
        which then feeds market_stall and wholesale. The total downstream
        potential is the weighted sum of all reachable task energies.

        Uses the transitive closure of the downstream graph (not just immediate
        children) to capture multi-hop value chains.
        """
        visited: set[str] = set()
        total = 0.0
        frontier = [(task_id, 1.0)]

        while frontier:
            current, coupling = frontier.pop(0)
            for downstream_id, coeff in snapshot.downstream.get(current, []):
                if downstream_id in visited:
                    continue
                visited.add(downstream_id)
                accumulated = coupling * coeff
                downstream_energy = snapshot.task_energy.get(downstream_id, 0.0)
                total += downstream_energy * accumulated
                frontier.append((downstream_id, accumulated))

        return total

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

            # ── Factor 1: Learned competition-adjusted expected energy ─
            # Use the learned competition pressure to discount expected payoff.
            # Tasks with high observed competition get their expected energy
            # reduced — the agent is less likely to get the full reward.
            pressure = self._competition_pressure.get(task_id, 0.0)
            expected_energy = energy / (1.0 + pressure * self.competition_weight)

            # ── Factor 2: Downstream multiplier ──────────────────────
            downstream_potential = self._downstream_energy_potential(task_id, snap)
            downstream_bonus = downstream_potential * self.downstream_weight * 0.01

            # ── Factor 3: Capability exclusivity ─────────────────────
            if task_caps:
                exclusivity = 1.0 / len(task_caps)
            else:
                exclusivity = 0.5
            exclusivity_bonus = (1.0 - exclusivity) * self.exclusivity_weight * 20

            score = expected_energy + downstream_bonus + exclusivity_bonus

            scores[task_id] = score
            reasoning_parts[task_id] = (
                f"E={energy:.0f}→exp={expected_energy:.1f}"
                f" (pressure={pressure:.2f})"
                f" + downstream={downstream_bonus:.1f}"
                f" + excl={exclusivity_bonus:.1f}"
                f" = {score:.1f}"
            )

        if not scores:
            return AgentDecision(task_id=None, reasoning="no viable tasks")

        # Softmax sampling over STRUCTURED scores.
        best_task = _softmax_sample(scores, temperature=1.0)

        return AgentDecision(
            task_id=best_task,
            expected_efficiency=self.capability.base_efficiency,
            reasoning=f"field: {reasoning_parts[best_task]}",
        )
