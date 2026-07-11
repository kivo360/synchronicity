"""
σ-aware agent — uses the σ framework's composition rule for scoring.

This agent replaces both the FieldAgent heuristic and the ActiveInferenceAgent
with scoring derived directly from the σ framework:

  score(task) = σ_chain(agent, task) × energy(task)

Where σ_chain is the composition σ₁·σ₂·...·σₙ along the downstream value
chain (the multiplication rule from the Mathematics 2026 paper).

This is the principled replacement for the hand-tuned downstream_weight.
Instead of `energy + downstream_potential × 0.5 × 0.01`, it's:

  expected_yield = energy × σ_direct × σ_downstream₁ × σ_downstream₂ × ...

The agent picks tasks that maximize expected total yield through the chain,
accounting for the fact that each hop loses efficiency multiplicatively.

Additional σ-framework signals:
  - Competition: tasks where other agents have low σ are opportunities
    (the agent can do better). Tasks where others have high σ are
    competitive (less advantage).
  - Temperature: T = 1/(1-σ) — high-T tasks are "hot" (high-efficiency
    agents doing easy work). The agent preferentially targets tasks
    where its own T is highest.

§12 connection: the learning-aware scoring function makes the agent
strategically BUILD σ through practice — the same process as LLM training
building prediction capability through gradient descent. More context
(completed tasks) = higher σ = cheaper future predictions = more value.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from synchronicity.field import FieldSnapshot
from synchronicity.value_chain import ValueChain
from synchronicity.sigma_framework import SigmaTracker
from synchronicity.agents import (
    AgentCapability, AgentDecision, BaseAgent, _softmax_sample,
)

logger = logging.getLogger(__name__)


class SigmaAgent(BaseAgent):
    """Agent that scores tasks using the σ composition rule + learning foresight.

    The scoring function is:

      score(task) = immediate_value(task) + future_value(task)

    Where:
      immediate_value = energy × σ_chain × temperature_factor

      future_value = (Δσ × σ_chain_new × expected_future_energy) × discount

    The future_value term is what makes this agent STRATEGIC about learning.
    It doesn't just maximize current yield — it maximizes the stream of
    future yields that become available as σ grows through practice.

    This is the §12 connection: the agent's σ growth IS context accumulation.
    Like an LLM that gets better at prediction with more training data, the
    agent gets better at a task with more practice. The learning-aware agent
    strategically invests in σ growth by targeting tasks where the expertise
    gain creates the highest long-term return.

    A task where the agent is already expert (σ=0.95) has low Δσ — little
    room to improve. A task where the agent is mediocre (σ=0.80) but does
    it frequently (high energy throughput) has high future value — the
    expertise gain compounds across many future completions.
    """

    def __init__(
        self,
        capability: AgentCapability,
        chain: ValueChain,
        sigma_tracker: SigmaTracker,
        learning_foresight: float = 0.5,
        discount_rate: float = 0.95,
    ):
        super().__init__(capability)
        self.chain = chain
        self.sigma_tracker = sigma_tracker
        self.learning_foresight = learning_foresight
        self.discount_rate = discount_rate

        # Competition learning (same as other agents)
        self._competition_pressure: dict[str, float] = {}
        self._last_energy_observed: dict[str, float] = {}

        # Track which tasks we've done (for learning signal)
        self._my_task_counts: dict[str, int] = {}

    def observe(self, snapshot: FieldSnapshot) -> None:
        """Update competition model and store snapshot."""
        if self._last_energy_observed:
            for task_id, current_energy in snapshot.task_energy.items():
                prev = self._last_energy_observed.get(task_id, 0.0)
                drop = prev - current_energy
                if drop > 5.0:
                    old = self._competition_pressure.get(task_id, 0.0)
                    total_e = max(sum(snapshot.task_energy.values()), 1.0)
                    intensity = min(drop / total_e * 10, 1.0)
                    self._competition_pressure[task_id] = old + 0.15 * (intensity - old)
                elif drop <= 0:
                    old = self._competition_pressure.get(task_id, 0.0)
                    self._competition_pressure[task_id] = old * 0.925

        self._last_energy_observed = dict(snapshot.task_energy)
        self._last_snapshot = snapshot

    def receive_reward(self, reward: float, success: bool) -> None:
        """Track which tasks we've done for learning signal."""
        super().receive_reward(reward, success)
        if hasattr(self, '_last_task_id') and self._last_task_id and success:
            self._my_task_counts[self._last_task_id] = self._my_task_counts.get(self._last_task_id, 0) + 1

    def _compute_immediate_value(
        self, task_id: str, energy: float, snap: FieldSnapshot,
    ) -> tuple[float, dict]:
        """Immediate yield: energy × σ_chain × temperature factor."""
        sigma_chain = self.sigma_tracker.chain_sigma(
            self.id, task_id, self.chain, snap, max_depth=3,
        )
        pressure = self._competition_pressure.get(task_id, 0.0)
        effective_energy = energy / (1.0 + pressure)
        sigma_direct = self.sigma_tracker.get_sigma(self.id, task_id)
        T = self.sigma_tracker.get_temperature(self.id, task_id)

        value = effective_energy * sigma_chain
        value *= (0.5 + 0.5 * min(T / 10.0, 1.0))

        return value, {
            "sigma_chain": sigma_chain,
            "sigma_direct": sigma_direct,
            "T": T,
            "pressure": pressure,
            "effective_energy": effective_energy,
        }

    def _compute_future_value(
        self, task_id: str, energy: float, snap: FieldSnapshot,
    ) -> tuple[float, dict]:
        """Future value from σ growth through practice.

        This is the §12 learning foresight term. When the agent does this
        task, its σ goes up (expertise accumulation). Higher σ means:
          1. Less waste on future completions of this task
          2. More angel energy generated (α·đN·σ)
          3. Higher σ_chain for downstream scoring

        The future value is the discounted stream of additional yield
        from having a higher σ on this task and its downstream chain.

        Key insight: tasks where the agent has LOW current σ have HIGH
        learning potential (big Δσ). The agent should strategically invest
        in building expertise — just like an LLM benefits from training on
        domains where it's weak (higher gradient signal).
        """
        sigma_current = self.sigma_tracker.get_sigma(self.id, task_id)

        # Expertise rate: how much σ improves per completion
        expertise_rate = 0.02

        # How much σ will improve: bigger when current σ is low
        # σ_new = min(σ * (1 + rate/max(σ, 0.1)), 0.999)
        sigma_growth_factor = expertise_rate / max(sigma_current, 0.1)
        sigma_after = min(sigma_current * (1.0 + sigma_growth_factor), 0.999)
        delta_sigma = sigma_after - sigma_current

        # If σ is already near max, learning value is tiny
        if delta_sigma < 0.0001:
            return 0.0, {"delta_sigma": 0.0, "sigma_after": sigma_after, "future_value": 0.0}

        # Expected future yield improvement per completion:
        # Each future completion of this task saves energy × Δσ in waste.
        # Over N future completions with geometric discount:
        #   total_savings ≈ energy × Δσ × Σ γ^i = energy × Δσ / (1 - discount)
        n_downstream = len(snap.downstream.get(task_id, []))
        # Tasks with more downstream connections generate more recurring work
        expected_completions = max(10, 30 + n_downstream * 15)

        # Geometric series: sum of discount^i for i=0..N
        # ≈ 1/(1-discount) for large N, capped
        discount_sum = min(1.0 / (1.0 - self.discount_rate), expected_completions)

        # Future value = energy saved per completion × number of future completions
        # Scale by Δσ relative to current σ (bigger gains matter more)
        future_value = energy * delta_sigma * discount_sum * self.learning_foresight

        return future_value, {
            "delta_sigma": delta_sigma,
            "sigma_after": sigma_after,
            "discount_sum": discount_sum,
            "future_value": future_value,
        }

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

            # ── Immediate value (current yield) ──────────────────
            imm_value, imm_info = self._compute_immediate_value(task_id, energy, snap)

            # ── Future value (σ growth through practice) ─────────
            fut_value, fut_info = self._compute_future_value(task_id, energy, snap)

            score = imm_value + fut_value

            scores[task_id] = score
            reasoning_parts[task_id] = (
                f"imm={imm_value:.1f} (σ_chain={imm_info['sigma_chain']:.3f})"
                f" + fut={fut_value:.1f} (Δσ={fut_info['delta_sigma']:.4f})"
                f" = {score:.1f}"
            )

        if not scores:
            return AgentDecision(task_id=None, reasoning="no viable tasks")

        best_task = _softmax_sample(scores, temperature=1.0)

        # Track for learning signal
        self._last_task_id = best_task

        return AgentDecision(
            task_id=best_task,
            expected_efficiency=self.capability.base_efficiency,
            reasoning=f"sigma+learning: {reasoning_parts[best_task]}",
        )
