"""
Energy field: the dynamic state of capital/incentive flowing through a value chain.

This module implements the "energy field" — the dynamic counterpart to the
value chain topology. Where value_chain.py defines the geometry (which tasks
connect to which), this module tracks how much energy (capital/incentive) is
currently allocated to each task, and how that energy changes over time.

Physics analogy:
    - Value chain topology = metric tensor g_μν (the geometry of spacetime)
    - Energy field state   = stress-energy tensor T_μν (matter/energy distribution)
    - Reward machine       = field equations G_μν = κT_μν (the update rule)

Conservation law (first law of thermodynamics):
    Energy is neither created nor destroyed. The total energy in the system
    changes ONLY through:
      1. External injection (new capital enters the system)
      2. External extraction (capital leaves as realized profit)
      3. Entropy production (waste — irrecoverable energy loss)

    total_energy_before - total_energy_after - extracted - waste = 0

    This invariant is asserted on every field update via check_conservation().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field
from typing import Optional

import numpy as np

from synchronicity.value_chain import ValueChain

logger = logging.getLogger(__name__)

# Numerical tolerance for conservation checks. Floating-point arithmetic can't
# be exact, so we allow a small epsilon. If energy conservation is violated
# beyond this threshold, the formalization has a bug.
ENERGY_EPSILON = 1e-6


@dataclass
class FieldSnapshot:
    """An immutable point-in-time view of the energy field.

    This is what an agent sees when it reads the field. It contains
    everything the agent needs to make a decision about which task to pick:
    the available tasks, their current incentive levels, and the downstream
    value chain structure.

    Agents reason about this structurally — not as a scalar to maximize, but
    as a landscape to navigate.
    """

    # task_id → current energy allocation (the incentive to do this task)
    task_energy: dict[str, float]

    # task_id → list of (downstream_task_id, coupling_coefficient)
    downstream: dict[str, list[tuple[str, float]]]

    # task_id → task type (for capability matching)
    task_types: dict[str, str]

    # task_id → required capabilities
    task_capabilities: dict[str, frozenset[str]]

    # Current total energy and entropy in the system
    total_energy: float
    total_entropy: float
    tick: int

    def top_tasks(self, n: int = 5) -> list[tuple[str, float]]:
        """Return the n tasks with highest current energy."""
        return sorted(self.task_energy.items(), key=lambda x: -x[1])[:n]


class ConservationError(AssertionError):
    """Raised when energy conservation is violated.

    This is not a runtime error — it's a formalization error. If this fires,
    the physics is wrong, not the code logic. Go back to the math.
    """

    pass


class EnergyField:
    """The dynamic energy state of a value chain.

    The field tracks:
      - Per-task energy allocation (how much capital/incentive is directed at
        each task right now)
      - Cumulative entropy (wasted energy — the second law says this only
        increases)
      - Cumulative extracted energy (realized profit — energy that left the
        system as useful work)

    The field does NOT update itself. Updates come from the RewardMachine,
    which applies conservation-law-checked update rules.
    """

    def __init__(self, chain: ValueChain, initial_energy: float = 1000.0):
        self.chain = chain
        self.tick = 0

        # Per-task energy allocation. Initialized evenly across all tasks.
        n_tasks = len(chain)
        if n_tasks == 0:
            raise ValueError("Cannot create field over an empty value chain")

        per_task = initial_energy / n_tasks
        self._energy: dict[str, float] = {}
        for task_id in chain.all_task_ids():
            self._energy[task_id] = per_task

        # Bookkeeping
        self._total_injected: float = initial_energy
        self._total_extracted: float = 0.0
        self._total_waste: float = 0.0  # entropy (second law: only increases)

        # History for visualization and debugging
        self._history: list[dict[str, float]] = []
        self._entropy_history: list[float] = []
        self._energy_total_history: list[float] = []
        self._snapshot_history()

    # ── Reading ──────────────────────────────────────────────────────────

    def energy_at(self, task_id: str) -> float:
        """Current energy allocated to a task."""
        return self._energy.get(task_id, 0.0)

    def total_energy(self) -> float:
        """Sum of all task energies — the energy currently 'in play'."""
        return sum(self._energy.values())

    @property
    def total_injected(self) -> float:
        return self._total_injected

    @property
    def total_extracted(self) -> float:
        return self._total_extracted

    @property
    def total_waste(self) -> float:
        return self._total_waste

    @property
    def total_entropy(self) -> float:
        """Alias for total_waste — entropy IS waste energy."""
        return self._total_waste

    def snapshot(self) -> FieldSnapshot:
        """Capture an immutable view for agents to reason about."""
        return FieldSnapshot(
            task_energy=dict(self._energy),
            downstream={
                tid: self.chain.downstream(tid) for tid in self._energy
            },
            task_types={
                tid: self.chain.get_task(tid).task_type.value
                for tid in self._energy
            },
            task_capabilities={
                tid: self.chain.get_task(tid).required_capabilities
                for tid in self._energy
            },
            total_energy=self.total_energy(),
            total_entropy=self._total_waste,
            tick=self.tick,
        )

    # ── Mutation (called only by RewardMachine) ──────────────────────────

    def _inject(self, task_id: str, amount: float) -> None:
        """Add energy to a task (external capital entering the system).

        If the task is at capacity, only the accepted amount counts as
        injected — the overflow is simply not added (it stays "outside"
        the system). This preserves conservation: we only book what entered.
        """
        if amount < 0:
            raise ValueError("Cannot inject negative energy — use _extract")
        task = self.chain.get_task(task_id)
        available_capacity = task.energy_capacity - self._energy[task_id]
        actual = min(amount, max(available_capacity, 0.0))
        self._energy[task_id] += actual
        self._total_injected += actual

    def _extract(self, task_id: str, amount: float) -> None:
        """Remove energy from a task (realized profit leaving the system)."""
        if amount < 0:
            raise ValueError("Cannot extract negative energy — use _inject")
        if amount > self._energy[task_id] + ENERGY_EPSILON:
            raise ValueError(
                f"Cannot extract {amount} from task '{task_id}' "
                f"(has {self._energy[task_id]:.4f})"
            )
        self._energy[task_id] -= amount
        self._total_extracted += amount

    def _propagate(self, from_id: str, to_id: str, amount: float) -> float:
        """Move energy from one task to another along a coupling.

        This is the core of value chain formation: completing task A drains
        its energy and pushes some downstream to task B.

        If the target task is at capacity, the overflow is NOT dropped (that
        would violate conservation). It stays on the source task so the caller
        can route it elsewhere (e.g. extract as realized value).

        Returns:
            The amount of energy that was actually accepted by the target
            (may be less than `amount` if the target was near capacity).
        """
        if amount < 0:
            raise ValueError("Cannot propagate negative energy")
        if amount > self._energy[from_id] + ENERGY_EPSILON:
            raise ValueError(
                f"Cannot propagate {amount} from '{from_id}' "
                f"(has {self._energy[from_id]:.4f})"
            )
        task = self.chain.get_task(to_id)
        available_capacity = task.energy_capacity - self._energy[to_id]
        actual = min(amount, max(available_capacity, 0.0))
        self._energy[from_id] -= actual
        self._energy[to_id] += actual
        return actual

    def _dissipate(self, task_id: str, amount: float) -> None:
        """Energy lost as waste/entropy (transaction cost, inefficiency).

        The second law: this energy is irrecoverable. It's tracked separately
        and can only increase.
        """
        if amount < 0:
            raise ValueError("Cannot dissipate negative energy")
        if amount > self._energy[task_id] + ENERGY_EPSILON:
            raise ValueError(
                f"Cannot dissipate {amount} from '{task_id}' "
                f"(has {self._energy[task_id]:.4f})"
            )
        self._energy[task_id] -= amount
        self._total_waste += amount

    def _drain_all(self, task_id: str) -> float:
        """Remove and return ALL energy from a task."""
        energy = self._energy[task_id]
        self._energy[task_id] = 0.0
        return energy

    def _advance_tick(self) -> None:
        self.tick += 1
        self._snapshot_history()

    # ── Conservation checking ────────────────────────────────────────────

    def check_conservation(self) -> None:
        """Verify the first law of thermodynamics holds.

        total_injected = total_energy_in_field + total_extracted + total_waste

        If this fails, the formalization is broken. The simulation must stop.
        """
        energy_in_field = self.total_energy()
        accounted = energy_in_field + self._total_extracted + self._total_waste
        discrepancy = abs(accounted - self._total_injected)

        if discrepancy > ENERGY_EPSILON:
            raise ConservationError(
                f"ENERGY CONSERVATION VIOLATED at tick {self.tick}\n"
                f"  injected:    {self._total_injected:.6f}\n"
                f"  in field:    {energy_in_field:.6f}\n"
                f"  extracted:   {self._total_extracted:.6f}\n"
                f"  waste:       {self._total_waste:.6f}\n"
                f"  accounted:   {accounted:.6f}\n"
                f"  discrepancy: {discrepancy:.6f} (tolerance: {ENERGY_EPSILON})\n"
                f"This is a formalization error, not a runtime bug. "
                f"Check the reward machine update rules."
            )

    # ── History / visualization ──────────────────────────────────────────

    def _snapshot_history(self) -> None:
        self._history.append(dict(self._energy))
        self._entropy_history.append(self._total_waste)
        self._energy_total_history.append(self.total_energy())

    @property
    def history(self) -> list[dict[str, float]]:
        return self._history

    @property
    def entropy_history(self) -> list[float]:
        return self._entropy_history

    @property
    def energy_total_history(self) -> list[float]:
        return self._energy_total_history

    def summary(self) -> str:
        return (
            f"EnergyField(tick={self.tick}, "
            f"energy={self.total_energy():.2f}, "
            f"entropy={self._total_waste:.2f}, "
            f"extracted={self._total_extracted:.2f})"
        )
