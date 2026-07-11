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
"""

from __future__ import annotations

import logging
from typing import Optional

from synchronicity.field import FieldSnapshot
from synchronicity.value_chain import ValueChain
from synchronicity.sigma_framework import SigmaTracker
from synchronicity.agents import (
    AgentCapability, AgentDecision, BaseAgent, _softmax_sample,
)

logger = logging.getLogger(__name__)


class SigmaAgent(BaseAgent):
    """Agent that scores tasks using the σ composition rule.

    The scoring function IS the σ framework:

      score(task) = energy(task) × σ_chain(agent, task)

    Where σ_chain is the product of per-hop σ along the downstream value
    chain. This replaces ALL hand-tuned weights with a single principled
    quantity derived from the math.

    The agent also uses learned competition (same EMA over energy depletion
    as FieldAgent/ActiveInferenceAgent).
    """

    def __init__(
        self,
        capability: AgentCapability,
        chain: ValueChain,
        sigma_tracker: SigmaTracker,
    ):
        super().__init__(capability)
        self.chain = chain
        self.sigma_tracker = sigma_tracker

        # Competition learning (same as other agents)
        self._competition_pressure: dict[str, float] = {}
        self._last_energy_observed: dict[str, float] = {}

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

            # ── σ-chain score: energy × σ_chain ─────────────────
            # σ_chain accounts for downstream value chain efficiency
            sigma_chain = self.sigma_tracker.chain_sigma(
                self.id, task_id, self.chain, snap, max_depth=3,
            )

            # ── Competition adjustment ──────────────────────────
            # Tasks with high competition pressure get discounted
            pressure = self._competition_pressure.get(task_id, 0.0)
            effective_energy = energy / (1.0 + pressure)

            # ── Direct σ (agent's own efficiency at this task) ──
            sigma_direct = self.sigma_tracker.get_sigma(self.id, task_id)

            # ── Temperature (T = 1/(1-σ)) as opportunity signal ──
            # High T = agent is efficient at this task relative to complexity
            # Low T = agent struggles here
            T = self.sigma_tracker.get_temperature(self.id, task_id)

            score = effective_energy * sigma_chain

            # Temperature bonus: prefer tasks where agent runs "hot"
            score *= (0.5 + 0.5 * min(T / 10.0, 1.0))

            scores[task_id] = score
            reasoning_parts[task_id] = (
                f"E={energy:.0f}×σ_chain={sigma_chain:.3f}"
                f" (σ_direct={sigma_direct:.2f}, T={T:.1f}, pressure={pressure:.2f})"
                f" = {score:.1f}"
            )

        if not scores:
            return AgentDecision(task_id=None, reasoning="no viable tasks")

        best_task = _softmax_sample(scores, temperature=1.0)

        return AgentDecision(
            task_id=best_task,
            expected_efficiency=self.capability.base_efficiency,
            reasoning=f"sigma: {reasoning_parts[best_task]}",
        )
