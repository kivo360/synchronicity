"""
σ-based reward machine — replaces fixed entropy_rate/extraction_rate with
the σ framework's variable mechanism efficiency.

The original reward machine used two fixed scalars:
    entropy_rate = 0.05  (waste fraction per transaction)
    extraction_rate = 0.30 (agent's take rate)

The σ reward machine uses the equation of motion đN = σ·A:

    A  = energy at task (driving force)
    σ  = H(p)/H(p,q) (measured per agent-task pair by SigmaTracker)
    đN = σ·A (useful work)
    waste = A·(1-σ) (entropy — the second law)
    agent_share = đN · ρ (extraction rate, still global — this is the
                 platform's take rate, not the mechanism efficiency)
    propagated = đN · (1-ρ)

Conservation check: waste + agent_share + propagated = A·(1-σ) + σ·A = A ✓

The critical change: σ is NOT a global constant. A skilled agent (high σ)
on a simple task produces mostly useful work. An unskilled agent (low σ)
on a complex task produces mostly waste. Same task, same energy, different
outcome — because the agent is different.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field
from typing import Optional

from synchronicity.field import EnergyField
from synchronicity.value_chain import ValueChain
from synchronicity.sigma_framework import SigmaTracker, TaskComplexity, build_task_complexities
from synchronicity.reward_machine import RewardMachine, FieldEvent, EventType

logger = logging.getLogger(__name__)


class SigmaRewardMachine(RewardMachine):
    """Reward machine using variable σ instead of fixed parameters.

    Drop-in replacement for the standard RewardMachine. Uses the same
    conservation laws and update structure, but replaces:

        waste = energy × entropy_rate         (FIXED)
        extracted = available × extraction_rate (FIXED)

    with:

        σ = sigma_tracker.get_sigma(agent_id, task_id)  (VARIABLE)
        useful_work = energy × σ                         (đN = σ·A)
        waste = energy × (1 - σ)                         (second law)
        extracted = useful_work × ρ                      (take rate)
        propagated = useful_work × (1 - ρ)

    Where ρ (rho) is the extraction/take rate — what fraction of useful
    work the agent keeps as reward. This IS still a global parameter, but
    it represents the PLATFORM's fee structure, not the mechanism efficiency.
    Different from entropy_rate because it doesn't represent waste — it
    represents how value is split between the agent and the downstream chain.
    """

    def __init__(
        self,
        chain: ValueChain,
        sigma_tracker: SigmaTracker,
        extraction_rate: float = 0.30,
    ):
        """Initialize the σ-based reward machine.

        Args:
            chain: The value chain topology.
            sigma_tracker: Tracks per-agent, per-task σ.
            extraction_rate: Agent take rate ρ (fraction of useful work
                the agent keeps). Default 0.30.
        """
        # Skip parent's fixed-parameter init for entropy/extraction rates
        self.chain = chain
        self.sigma_tracker = sigma_tracker
        self.extraction_rate = extraction_rate
        self.agent_rewards: dict[str, float] = {}

        if not (0 <= extraction_rate <= 1):
            raise ValueError("extraction_rate must be in [0, 1]")

    def apply(self, field: EnergyField, event: FieldEvent) -> float:
        """Apply an event using variable σ.

        This overrides the parent's apply() to use the σ framework's
        equation of motion instead of fixed parameters.
        """
        if event.event_type == EventType.TASK_COMPLETED:
            reward = self._handle_completion(field, event)
        elif event.event_type == EventType.TASK_FAILED:
            reward = self._handle_failure(field, event)
        elif event.event_type == EventType.CAPITAL_INJECTED:
            field._inject(event.task_id, event.amount)
            reward = 0.0
        elif event.event_type == EventType.CAPITAL_EXTRACTED:
            field._extract(event.task_id, event.amount)
            reward = event.amount
        else:
            reward = 0.0

        # CRITICAL: verify conservation after every update
        field.check_conservation()

        # Update σ tracker with the observed outcome
        if event.event_type in (EventType.TASK_COMPLETED, EventType.TASK_FAILED):
            self.sigma_tracker.update(
                event.agent_id, event.task_id,
                success=(event.event_type == EventType.TASK_COMPLETED),
                efficiency=event.efficiency,
            )

        # Track rewards
        if event.agent_id and reward > 0:
            self.agent_rewards[event.agent_id] = (
                self.agent_rewards.get(event.agent_id, 0.0) + reward
            )

        field._advance_tick()
        return reward

    def _handle_completion(self, field: EnergyField, event: FieldEvent) -> float:
        """Process task completion using đN = σ·A.

        Energy flow:
          1. Get σ for this agent-task pair (variable, learned)
          2. đN = σ·A (useful work)
          3. waste = A·(1-σ) (entropy — second law, built in algebraically)
          4. agent_share = đN × ρ (agent reward)
          5. propagated = đN × (1-ρ) (downstream flow)
        """
        task_id = event.task_id
        agent_id = event.agent_id
        total_energy = field.energy_at(task_id)

        if total_energy <= 1e-10:
            return 0.0

        # ── Get variable σ ──────────────────────────────────────
        # Factor in efficiency (execution quality)
        sigma_base = self.sigma_tracker.get_sigma(agent_id, task_id)
        sigma_effective = sigma_base * event.efficiency
        sigma_effective = max(0.01, min(sigma_effective, 0.99))

        # ── đN = σ·A (useful work) ──────────────────────────────
        # The second law is built in: waste = A·(1-σ)
        e_waste = total_energy * (1.0 - sigma_effective)
        e_useful = total_energy * sigma_effective

        field._dissipate(task_id, e_waste)

        # ── Agent extracts reward ───────────────────────────────
        e_extract = e_useful * self.extraction_rate
        field._extract(task_id, e_extract)

        # ── Propagate downstream ────────────────────────────────
        e_remaining = e_useful - e_extract
        downstream = self.chain.downstream(task_id)

        if downstream:
            total_coeff = sum(coeff for _, coeff in downstream)
            if total_coeff <= 0:
                field._extract(task_id, e_remaining)
                e_extract += e_remaining
            else:
                for target_id, coeff in downstream:
                    normalized = coeff / total_coeff
                    e_to_prop = e_remaining * normalized
                    actual = field._propagate(task_id, target_id, e_to_prop)
                    overflow = e_to_prop - actual
                    if overflow > 1e-10:
                        field._extract(task_id, overflow)
                        e_extract += overflow
        else:
            field._extract(task_id, e_remaining)
            e_extract += e_remaining

        logger.info(
            f"[tick {field.tick}] {agent_id} completed {task_id}: "
            f"σ={sigma_effective:.3f} A={total_energy:.2f} → "
            f"đN={e_useful:.2f} waste={e_waste:.2f} "
            f"reward={e_extract:.2f}"
        )

        return e_extract

    def _handle_failure(self, field: EnergyField, event: FieldEvent) -> float:
        """Failed task attempt — high entropy, no useful work.

        A failure means σ → 0 for this attempt. All energy is wasted
        (maximum entropy production), and the task energy is partially
        dissipated.

        However, the σ tracker learns from failures — next time the agent
        attempts this task, its σ will be lower (higher H(p,q)).
        """
        task_id = event.task_id
        current_energy = field.energy_at(task_id)

        if current_energy <= 1e-10:
            return 0.0

        # Failure = very low σ → mostly waste
        sigma_failure = 0.05  # almost all energy wasted
        e_waste = current_energy * (1.0 - sigma_failure)
        field._dissipate(task_id, e_waste)

        logger.info(
            f"[tick {field.tick}] {event.agent_id} FAILED {task_id}: "
            f"wasted {e_waste:.2f} energy, "
            f"{field.energy_at(task_id):.2f} remains"
        )

        return 0.0
