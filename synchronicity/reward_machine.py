"""
Reward machine: the field update rules.

This module implements the dynamics — how the energy field changes when agents
act. It's the bridge between the static topology (value chain) and the dynamic
state (energy field).

In physics terms, this is the field equation: given the current state of the
field and an event (agent action), compute the new field state subject to:

  1. Conservation of energy (first law):
     No energy is created or destroyed. When a task is completed, its energy
     flows downstream along couplings, some is extracted as realized value,
     and some is dissipated as entropy (waste).

  2. Entropy increase (second law):
     Every transaction has a cost. The entropy fraction τ ∈ [0, 1) represents
     the irrecoverable energy loss per interaction. Total entropy can only
     increase. It can never decrease.

  3. Capacity saturation:
     No task node can hold more energy than its energy_capacity. Overflow
     during propagation is extracted as realized value (the task produced more
     than it could absorb — excess returns to the system as profit).

The update sequence for a task completion event:

    [Task A completes with energy E]
       │
       ├─ 1. Entropy:      E_waste = E × τ         (waste, irrecoverable)
       ├─ 2. Available:    E_avail = E - E_waste    (what's left to route)
       ├─ 3. Extracted:    E_extract = E_avail × ρ  (agent's realized reward)
       ├─ 4. Propagated:   E_prop = E_avail - E_extract
       │      └─ split across downstream couplings by normalized coefficient
       └─ 5. Overflow:     any downstream capacity overflow → extracted

Where:
    τ (tau)    = entropy rate — transaction cost fraction per completion
    ρ (rho)    = extraction rate — agent's share of available energy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Optional

from synchronicity.field import EnergyField
from synchronicity.value_chain import ValueChain, TaskType

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of events that can trigger a field update."""

    TASK_COMPLETED = "task_completed"    # agent finished a task
    TASK_FAILED = "task_failed"          # agent attempted but failed
    CAPITAL_INJECTED = "capital_injected"  # external money entered
    CAPITAL_EXTRACTED = "capital_extracted"  # external money removed
    AGENT_ARRIVAL = "agent_arrival"      # new agent joined (adds capability)
    AGENT_DEPARTURE = "agent_departure"  # agent left (may drain orphaned energy)


@dataclass
class FieldEvent:
    """An event that triggers a field update.

    Attributes:
        event_type: What happened.
        task_id: Which task the event concerns (required for most events).
        agent_id: Which agent caused the event (optional).
        amount: Magnitude (e.g. capital amount for injection events).
        efficiency: How well the agent performed (0.0 = total failure,
            1.0 = perfect execution). Affects how much energy propagates
            downstream vs. is dissipated as additional entropy.
        metadata: Free-form dict for scenario-specific annotations.
    """

    event_type: EventType
    task_id: str = ""
    agent_id: str = ""
    amount: float = 0.0
    efficiency: float = 1.0
    metadata: dict = dc_field(default_factory=dict)

    def __post_init__(self):
        if not (0.0 <= self.efficiency <= 1.0):
            raise ValueError(
                f"efficiency must be in [0, 1], got {self.efficiency}"
            )


class RewardMachine:
    """Applies conservation-law-checked updates to the energy field.

    The reward machine is the ONLY thing allowed to mutate the energy field.
    It applies the update rules, routes energy through the value chain, and
    verifies conservation on every single update.

    Configuration:
        entropy_rate (τ): Fraction of energy lost as waste per task completion.
            Default 0.05 (5%). This is the "transaction cost" — analogous to
            the friction in a physical system. Higher = more waste, faster
            energy depletion.

        extraction_rate (ρ): Fraction of available energy an agent extracts as
            its reward. Default 0.30 (30%). The rest propagates downstream.
            This is the agent's "take rate" for doing the work.

        fail_entropy_rate: Additional entropy penalty when an agent fails a
            task. Default 0.15. Failed work wastes more energy than successful
            work.
    """

    def __init__(
        self,
        chain: ValueChain,
        entropy_rate: float = 0.05,
        extraction_rate: float = 0.30,
        fail_entropy_rate: float = 0.15,
    ):
        self.chain = chain
        self.entropy_rate = entropy_rate
        self.extraction_rate = extraction_rate
        self.fail_entropy_rate = fail_entropy_rate

        if not (0 <= entropy_rate < 1):
            raise ValueError("entropy_rate must be in [0, 1)")
        if not (0 <= extraction_rate <= 1):
            raise ValueError("extraction_rate must be in [0, 1]")
        if not (0 <= fail_entropy_rate < 1):
            raise ValueError("fail_entropy_rate must be in [0, 1)")

        # Track total extracted per agent for reward accounting
        self.agent_rewards: dict[str, float] = {}

    def apply(self, field: EnergyField, event: FieldEvent) -> float:
        """Apply an event to the field and return the agent's reward.

        This is the main entry point. The simulation loop calls this when
        an agent completes/fails a task.

        Returns:
            The reward (extracted energy) for the agent, if any.
        """
        handler = {
            EventType.TASK_COMPLETED: self._handle_completion,
            EventType.TASK_FAILED: self._handle_failure,
            EventType.CAPITAL_INJECTED: self._handle_injection,
            EventType.CAPITAL_EXTRACTED: self._handle_extraction,
            EventType.AGENT_ARRIVAL: self._handle_agent_arrival,
            EventType.AGENT_DEPARTURE: self._handle_agent_departure,
        }.get(event.event_type)

        if handler is None:
            raise ValueError(f"Unknown event type: {event.event_type}")

        reward = handler(field, event)

        # CRITICAL: verify conservation after every single update.
        field.check_conservation()

        # Track rewards
        if event.agent_id and reward > 0:
            self.agent_rewards[event.agent_id] = (
                self.agent_rewards.get(event.agent_id, 0.0) + reward
            )

        field._advance_tick()
        return reward

    def _handle_completion(self, field: EnergyField, event: FieldEvent) -> float:
        """Process a successful task completion.

        Energy flow:
          1. Drain all energy from the completed task
          2. Dissipate entropy (transaction cost)
          3. Agent extracts its share (reward)
          4. Remaining energy propagates downstream along couplings
          5. Any downstream capacity overflow is extracted as additional reward

        Returns: the agent's reward (extracted energy).
        """
        task_id = event.task_id
        total_energy = field.energy_at(task_id)
        agent_id = event.agent_id

        if total_energy <= 1e-10:
            logger.debug(
                f"Task {task_id} completed but had no energy — no-op"
            )
            return 0.0

        # 1. Factor in agent efficiency — inefficient work wastes more
        effective_entropy = self.entropy_rate + (
            (1 - event.efficiency) * self.fail_entropy_rate
        )
        effective_entropy = min(effective_entropy, 0.95)  # safety clamp

        # 2. Dissipate entropy (second law — irrecoverable)
        e_waste = total_energy * effective_entropy
        field._dissipate(task_id, e_waste)

        e_available = total_energy - e_waste

        # 3. Agent extracts its reward
        e_extract = e_available * self.extraction_rate
        field._extract(task_id, e_extract)

        # 4. Propagate remaining energy downstream
        e_remaining = e_available - e_extract
        downstream = self.chain.downstream(task_id)

        if downstream:
            # Normalize coefficients so they sum to 1 across all edges
            total_coeff = sum(coeff for _, coeff in downstream)
            if total_coeff <= 0:
                # Degenerate — just extract remaining as profit
                field._extract(task_id, e_remaining)
                e_extract += e_remaining
            else:
                for target_id, coeff in downstream:
                    normalized = coeff / total_coeff
                    e_to_propagate = e_remaining * normalized
                    actual = field._propagate(task_id, target_id, e_to_propagate)
                    # Energy the target couldn't absorb → extract as realized value
                    overflow = e_to_propagate - actual
                    if overflow > 1e-10:
                        field._extract(task_id, overflow)
                        e_extract += overflow
        else:
            # No downstream — terminal task. All remaining energy is extracted.
            field._extract(task_id, e_remaining)
            e_extract += e_remaining

        logger.info(
            f"[tick {field.tick}] {agent_id} completed {task_id}: "
            f"E={total_energy:.2f} → waste={e_waste:.2f}, "
            f"reward={e_extract:.2f}, propagated={e_remaining if downstream else 0:.2f}"
        )

        return e_extract

    def _handle_failure(self, field: EnergyField, event: FieldEvent) -> float:
        """Process a failed task attempt.

        Energy isn't fully lost — the task remains incomplete — but a
        significant fraction is dissipated as entropy (wasted effort).
        The remaining energy stays in the task for the next agent.

        Returns: 0 (agents don't get rewarded for failures).
        """
        task_id = event.task_id
        current_energy = field.energy_at(task_id)

        if current_energy <= 1e-10:
            return 0.0

        # Failed work wastes more energy than the base entropy rate
        e_waste = current_energy * self.fail_entropy_rate
        field._dissipate(task_id, e_waste)

        logger.info(
            f"[tick {field.tick}] {event.agent_id} FAILED {task_id}: "
            f"wasted {e_waste:.2f} energy, "
            f"{field.energy_at(task_id):.2f} remains"
        )

        return 0.0

    def _handle_injection(self, field: EnergyField, event: FieldEvent) -> float:
        """External capital enters the system at a specific task."""
        field._inject(event.task_id, event.amount)
        logger.info(
            f"[tick {field.tick}] CAPITAL INJECTED: "
            f"{event.amount:.2f} → task '{event.task_id}'"
        )
        return 0.0

    def _handle_extraction(self, field: EnergyField, event: FieldEvent) -> float:
        """External capital leaves the system (e.g. investor withdraws)."""
        field._extract(event.task_id, event.amount)
        return event.amount

    def _handle_agent_arrival(self, field: EnergyField, event: FieldEvent) -> float:
        """Agent arrival — currently informational, no energy impact.

        In future versions, agent arrival could boost the energy capacity of
        tasks matching their capabilities (more supply = more potential).
        """
        logger.debug(
            f"[tick {field.tick}] Agent {event.agent_id} arrived"
        )
        return 0.0

    def _handle_agent_departure(self, field: EnergyField, event: FieldEvent) -> float:
        """Agent departure — currently informational.

        In future versions, agent departure could reduce capability availability,
        potentially stranding energy in tasks nobody can do.
        """
        logger.debug(
            f"[tick {field.tick}] Agent {event.agent_id} departed"
        )
        return 0.0
