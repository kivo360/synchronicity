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
    uses three strategic signals that greedy ignores:

    1. Competition estimation: The expected value of targeting a task drops
       as more agents in the same batch can also target it. If 3 laborers
       all see collect_north with 80 energy, each one's expected payoff is
       ~27, not 80 — because only one of them will get it. Greedy agents
       all rush the same task and collide. Field agents spread out.

    2. Downstream multiplier: Completing a bridge task unlocks energy
       downstream. The true value of sorting isn't just its own energy —
       it's its energy PLUS the downstream energy it propagates to (weighted
       by coupling coefficients). A greedy agent sees sorting as "just
       another task." A field agent sees it as a multiplier.

    3. Capability exclusivity: If the agent is one of few who can do a
       bottleneck task (like sorting, which only specialists can do), the
       system needs them there. The field agent values tasks higher when
       fewer competitors can do them.

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
    ):
        super().__init__(capability)
        self.chain = chain
        self.downstream_weight = downstream_weight
        self.competition_weight = competition_weight
        self.exclusivity_weight = exclusivity_weight
        # Track which agents we've observed (for competition estimation)
        self._observed_agents: set[str] = set()

    def observe(self, snapshot: FieldSnapshot) -> None:
        """Track snapshot for decision-making."""
        self._last_snapshot = snapshot

    def _estimate_competition(
        self, task_id: str, snapshot: FieldSnapshot
    ) -> float:
        """Estimate how many OTHER agents will likely target this task.

        We don't know exactly what other agents will do, but we can estimate
        based on: how many distinct capability sets could target this task?
        In a batch of N agents, if K agents have matching capabilities and
        the task has high energy, expect ~min(K, batch_size) competitors.

        Since the agent doesn't know the exact batch composition, we use a
        heuristic: tasks with common capabilities (like "labor") face more
        competition than tasks with rare capabilities (like "sorting").

        Returns: estimated number of competing agents (excluding self).
        """
        task_caps = snapshot.task_capabilities.get(task_id, frozenset())
        if not task_caps:
            return 0.0  # anyone can do it → high competition, handled below

        # How common are the required capabilities?
        # We infer this from the task: capabilities that appear on many tasks
        # are common (labor), capabilities that appear on few tasks are rare
        # (sorting, processing, sales).
        cap_frequency: dict[str, int] = {}
        for tid, caps in snapshot.task_capabilities.items():
            for cap in caps:
                cap_frequency[cap] = cap_frequency.get(cap, 0) + 1

        # Average frequency of this task's required capabilities.
        # High frequency = common skill = many potential competitors.
        avg_freq = sum(cap_frequency.get(c, 1) for c in task_caps) / len(task_caps)
        n_tasks = len(snapshot.task_capabilities)

        # Normalize: if the capability appears on most tasks, competition is
        # high. If it's rare, competition is low.
        commonness = avg_freq / max(n_tasks, 1)
        # Scale to an estimated competitor count (heuristic)
        estimated_competitors = commonness * 4.0  # tuneable scaling

        return estimated_competitors

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
        # BFS/DFS through downstream graph, accumulating weighted energy
        visited: set[str] = set()
        total = 0.0
        frontier = [(task_id, 1.0)]  # (task_id, accumulated_coupling)

        while frontier:
            current, coupling = frontier.pop(0)
            for downstream_id, coeff in snapshot.downstream.get(current, []):
                if downstream_id in visited:
                    continue
                visited.add(downstream_id)
                accumulated = coupling * coeff
                downstream_energy = snapshot.task_energy.get(downstream_id, 0.0)
                total += downstream_energy * accumulated
                # Add capacity as potential (even if currently empty,
                # completing this task could fill it)
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

            # ── Factor 1: Competition-adjusted expected energy ───────
            # The agent won't get the full energy if other agents are also
            # targeting this task. Expected value = energy / (1 + competitors)
            competitors = self._estimate_competition(task_id, snap)
            expected_energy = energy / (1.0 + competitors * self.competition_weight)

            # ── Factor 2: Downstream multiplier ──────────────────────
            # The true value of this task includes the downstream energy it
            # unlocks. This is what makes bridge tasks (like sorting) worth
            # more than their raw energy suggests.
            downstream_potential = self._downstream_energy_potential(task_id, snap)
            downstream_bonus = downstream_potential * self.downstream_weight * 0.01

            # ── Factor 3: Capability exclusivity ─────────────────────
            # If few tasks require this agent's capabilities, the agent is
            # more valuable at this task. A specialist doing sorting (which
            # only 2 agents can do) is more valuable than a laborer collecting
            # (which 3 agents can do).
            if task_caps:
                exclusivity = 1.0 / len(task_caps)  # simpler requirement = less exclusive
            else:
                exclusivity = 0.5  # no requirements = anyone can do it
            exclusivity_bonus = (1.0 - exclusivity) * self.exclusivity_weight * 20

            score = expected_energy + downstream_bonus + exclusivity_bonus

            scores[task_id] = score
            reasoning_parts[task_id] = (
                f"E={energy:.0f}→exp={expected_energy:.1f}"
                f" + downstream={downstream_bonus:.1f}"
                f" + excl={exclusivity_bonus:.1f}"
                f" = {score:.1f}"
            )

        if not scores:
            return AgentDecision(task_id=None, reasoning="no viable tasks")

        # Softmax sampling over STRUCTURED scores. Same selection mechanism as
        # GreedyAgent, but the scores encode competition-adjusted expected
        # energy, downstream multiplier, and capability exclusivity.
        best_task = _softmax_sample(scores, temperature=1.0)

        return AgentDecision(
            task_id=best_task,
            expected_efficiency=self.capability.base_efficiency,
            reasoning=f"field: {reasoning_parts[best_task]}",
        )
