"""
Angel coefficient reward machine — endogenous gradient generation.

dA/dt = (α − γ)·σ·A

This is the most novel part of the Synchronicity σ framework. The standard
reward machine only drains energy — agents act, energy depletes, done.

The angel coefficient says: completing useful work CREATES new gradients.
When an agent completes a task with high σ, it doesn't just move energy
downstream — it reveals new opportunities. New energy appears at locations
determined by the agent's insight (σ) and the existing topology.

This makes the distribution ENDOGENOUS — acting on the field changes the
field. The system can grow (α > γ) or saturate (α < γ).

α (angel coefficient): ∂A/∂(đN) — rate at which completed work generates
    new driving force. High-σ agents create more gradients because they
    see deeper into the system.

γ (gradient consumption): rate at which existing gradients decay. Old
    opportunities fade as the environment changes. This is the reason
    every measured domain favors saturation — unused gradients erode.

The mechanism:
    On each task completion:
      1. Standard σ drain: đN = σ·A, waste = A·(1-σ)
      2. Angel injection: new_energy = α · đN · σ_informed
         Injected at DOWNSTREAM tasks, weighted by coupling coefficients.
         This represents the agent discovering new opportunities through
         the act of completing work.
      3. Gradient decay: every task's energy decays by γ each tick.
         A_energy *= (1 - γ). This represents opportunities fading.

Net effect: if α > γ, the system grows (more total energy over time).
          if α < γ, the system saturates (energy depletes despite work).
          if α = γ, the system is in equilibrium.

Conservation check MODIFIES to:
    injected + angel_generated = field_energy + extracted + waste + decayed
"""

from __future__ import annotations

import logging
from typing import Optional

from synchronicity.field import EnergyField
from synchronicity.value_chain import ValueChain
from synchronicity.sigma_framework import SigmaTracker
from synchronicity.sigma_reward_machine import SigmaRewardMachine
from synchronicity.reward_machine import FieldEvent, EventType

logger = logging.getLogger(__name__)


class AngelRewardMachine(SigmaRewardMachine):
    """Reward machine with endogenous gradient generation (angel coefficient).

    Extends the σ reward machine with two new dynamics:

    1. Angel injection: completing tasks generates new energy at downstream
       nodes. The amount depends on α (angel coefficient) and the agent's σ.
       A high-σ agent that completes work efficiently reveals MORE new
       opportunities than a low-σ agent doing the same task.

    2. Gradient decay: all task energy decays by factor γ each tick.
       Unused opportunities fade. This is the reason the paper says "every
       measured domain so far favors saturation."

    The conservation law is MODIFIED:
        injected + angel_generated = field + extracted + waste + decayed
    """

    def __init__(
        self,
        chain: ValueChain,
        sigma_tracker: SigmaTracker,
        extraction_rate: float = 0.30,
        angel_alpha: float = 0.0,
        gradient_gamma: float = 0.0,
    ):
        # Don't call parent's __init__ — we override it entirely
        self.chain = chain
        self.sigma_tracker = sigma_tracker
        self.extraction_rate = extraction_rate
        self.angel_alpha = angel_alpha
        self.gradient_gamma = gradient_gamma
        self.agent_rewards: dict[str, float] = {}

        # Track angel-generated energy for conservation
        self._total_angel_generated: float = 0.0
        self._total_decayed: float = 0.0

        if not (0 <= extraction_rate <= 1):
            raise ValueError("extraction_rate must be in [0, 1]")

    def apply(self, field: EnergyField, event: FieldEvent) -> float:
        """Apply event with angel coefficient dynamics."""
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

        # Update σ tracker
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

    def apply_tick_decay(self, field: EnergyField) -> None:
        """Apply per-tick gradient decay to all tasks.

        Must be called once per tick (after all agent actions, before
        conservation check). Reduces all task energy by factor γ,
        representing opportunities fading over time.
        """
        if self.gradient_gamma <= 0:
            return

        for task_id in field._energy:
            decay_amount = field._energy[task_id] * self.gradient_gamma
            if decay_amount > 0.01:
                field._dissipate(task_id, decay_amount)
                self._total_decayed += decay_amount

    def check_conservation(self, field: EnergyField) -> None:
        """Modified conservation: injected + angel = field + extracted + waste

        Note: decayed energy goes through field._dissipate() which adds to
        field._total_waste. So we DON'T add _total_decayed separately — it's
        already accounted for in waste.
        """
        energy_in_field = field.total_energy()
        accounted = (
            energy_in_field
            + field._total_extracted
            + field._total_waste
        )
        total_input = field._total_injected + self._total_angel_generated
        discrepancy = abs(accounted - total_input)

        if discrepancy > 0.1:
            from synchronicity.field import ConservationError
            raise ConservationError(
                f"ANGEL CONSERVATION VIOLATED at tick {field.tick}\n"
                f"  injected:         {field._total_injected:.4f}\n"
                f"  angel_generated:  {self._total_angel_generated:.4f}\n"
                f"  total input:      {total_input:.4f}\n"
                f"  in field:         {energy_in_field:.4f}\n"
                f"  extracted:        {field._total_extracted:.4f}\n"
                f"  waste:            {field._total_waste:.4f}\n"
                f"  accounted:        {accounted:.4f}\n"
                f"  discrepancy:      {discrepancy:.4f}\n"
            )

    def _handle_completion(self, field: EnergyField, event: FieldEvent) -> float:
        """Process completion with angel injection + σ drain."""
        task_id = event.task_id
        agent_id = event.agent_id
        total_energy = field.energy_at(task_id)

        if total_energy <= 1e-10:
            return 0.0

        # ── σ drain (same as SigmaRewardMachine) ─────────────────
        sigma_base = self.sigma_tracker.get_sigma(agent_id, task_id)
        sigma_effective = max(0.01, min(sigma_base * event.efficiency, 0.99))

        e_waste = total_energy * (1.0 - sigma_effective)
        e_useful = total_energy * sigma_effective

        field._dissipate(task_id, e_waste)

        e_extract = e_useful * self.extraction_rate
        field._extract(task_id, e_extract)

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

        # ── ANGEL INJECTION ──────────────────────────────────────
        # Completing work generates new gradients.
        # amount = α · đN · σ_agent (high-σ agents reveal more opportunities)
        # Injected at downstream tasks and the completed task itself.
        if self.angel_alpha > 0 and downstream:
            # The "informed σ" includes the dual effect: completing work
            # raised σ AND revealed new A. We approximate with current σ.
            angel_energy = self.angel_alpha * e_useful * sigma_effective

            # Distribute new energy: some stays at this task (refined
            # understanding creates new local opportunities), most goes
            # downstream (new value chain branches discovered).
            local_fraction = 0.2  # 20% stays local
            local_energy = angel_energy * local_fraction
            downstream_energy = angel_energy * (1.0 - local_fraction)

            # Inject local
            actual_local = self._angel_inject(field, task_id, local_energy)
            self._total_angel_generated += actual_local

            # Inject downstream proportional to coupling
            total_coeff = sum(coeff for _, coeff in downstream)
            if total_coeff > 0:
                for target_id, coeff in downstream:
                    share = downstream_energy * (coeff / total_coeff)
                    actual_ds = self._angel_inject(field, target_id, share)
                    self._total_angel_generated += actual_ds

        logger.info(
            f"[tick {field.tick}] {agent_id} completed {task_id}: "
            f"σ={sigma_effective:.3f} đN={e_useful:.2f} → "
            f"reward={e_extract:.2f}"
            + (f" angel={self.angel_alpha * e_useful * sigma_effective:.2f}"
               if self.angel_alpha > 0 else "")
        )

        return e_extract

    def _handle_failure(self, field: EnergyField, event: FieldEvent) -> float:
        """Failed attempt — high waste, no angel injection."""
        task_id = event.task_id
        current_energy = field.energy_at(task_id)

        if current_energy <= 1e-10:
            return 0.0

        sigma_failure = 0.05
        e_waste = current_energy * (1.0 - sigma_failure)
        field._dissipate(task_id, e_waste)

        return 0.0

    def _angel_inject(self, field: EnergyField, task_id: str, amount: float) -> float:
        """Inject angel-generated energy (only up to capacity).

        This energy is tracked separately from external injection in
        _total_angel_generated. It is NOT added to field._total_injected
        to avoid double-counting in conservation.
        """
        if amount <= 0:
            return 0.0
        task = self.chain.get_task(task_id)
        available = task.energy_capacity - field._energy[task_id]
        actual = min(amount, max(available, 0.0))
        field._energy[task_id] += actual
        return actual

    @property
    def total_angel_generated(self) -> float:
        return self._total_angel_generated

    @property
    def total_decayed(self) -> float:
        return self._total_decayed
