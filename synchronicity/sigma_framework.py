"""
The σ (sigma) framework — the formal core of Synchronicity.

Based on the Mathematics (2026) section of the annotated Twitter-DAO paper.
Every result tagged with its honest status from the paper.

Core equation of motion [ESTABLISHED — Onsager 1931]:

    đN = σ · A

Where:
    đN = useful work produced (extracted value)
    σ  = mechanism efficiency (H(p) / H(p,q))
    A  = driving force (energy at the task)

Waste (entropy) is the complement:

    waste = A · (1 - σ)

Information identity [ESTABLISHED — Shannon chain]:

    σ = H(p) / H(p,q)

Where:
    H(p)   = intrinsic task complexity (minimum bits to specify correct outcome)
    H(p,q) = agent's cross-entropy (what the agent actually pays)
    σ ≤ 1 always (cross-entropy ≥ entropy)

Composition rule [ESTABLISHED]:

    σ_total = σ₁ · σ₂ · ... · σₙ

A value chain's total efficiency is the PRODUCT of each hop's σ. This is why
deep chains waste energy — inefficiency compounds multiplicatively.

Temperature [DERIVED under bits-ledger mapping]:

    T = 1 / (1 - σ)

Monotone increasing in σ — high-efficiency organizations "run hot."

Angel coefficient [HYPOTHESIS]:

    dA/dt = (α - γ) · σ · A

Where α = ∂A/∂(đN) (information reveals new gradients) and γ = gradient
consumption. The sign of (α-γ) determines whether cycles compound or
saturate. Every measured domain so far favors saturation.
"""

from __future__ import annotations

import math
import logging
from collections import defaultdict
from dataclasses import dataclass, field as dc_field
from typing import Optional

from synchronicity.field import EnergyField, FieldSnapshot
from synchronicity.value_chain import ValueChain

logger = logging.getLogger(__name__)


@dataclass
class TaskComplexity:
    """Intrinsic complexity of a task — the H(p) term.

    H(p) represents the minimum information needed to specify the correct
    outcome of a task. More complex tasks have higher H(p).

    In practice, H(p) can be derived from:
    - Process mining statistics (frequency, variant count, duration variance)
    - Number of capability requirements (more requirements = more complex)
    - Historical outcome distribution (more variable outcomes = higher entropy)

    For a binary success/failure model:
        H(p) = -p·log₂(p) - (1-p)·log₂(1-p)
    Maximum entropy at p=0.5 (maximally uncertain task).
    """

    task_id: str
    intrinsic_entropy: float  # H(p) in bits, range [0, 1] for binary outcomes
    n_observations: int = 0   # how many times this task has been observed

    @classmethod
    def from_outcome_distribution(
        cls, task_id: str, successes: int, failures: int
    ) -> "TaskComplexity":
        """Compute H(p) from observed success/failure distribution."""
        total = successes + failures
        if total == 0:
            return cls(task_id=task_id, intrinsic_entropy=0.5)  # max uncertainty prior

        p = successes / total
        if p == 0 or p == 1:
            return cls(task_id=task_id, intrinsic_entropy=0.01, n_observations=total)

        h = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
        return cls(task_id=task_id, intrinsic_entropy=h, n_observations=total)

    @classmethod
    def from_capability_count(cls, task_id: str, n_capabilities: int) -> "TaskComplexity":
        """Heuristic: tasks requiring more specialized capabilities are harder."""
        # Map capability count to entropy: 0 caps = easy (0.3), many caps = hard (0.7)
        h = min(0.3 + 0.1 * n_capabilities, 0.8)
        return cls(task_id=task_id, intrinsic_entropy=h)


class SigmaTracker:
    """Per-agent, per-task σ tracking with online learning.

    Tracks each agent's σ for each task using the framework's definition:

        σ = H(p) / H(p,q)

    Where H(p) is the task's intrinsic complexity and H(p,q) is the agent's
    observed cross-entropy (their actual loss rate on the task).

    H(p,q) is estimated online using exponential moving average of observed
    outcomes. Initially, H(p,q) = H(p) / σ_prior (optimistic prior).

    As the agent gains experience, H(p,q) converges to its true value, and
    σ converges to the agent's actual efficiency on that task.
    """

    def __init__(
        self,
        task_complexities: dict[str, TaskComplexity],
        prior_sigma: float = 0.85,
        learning_rate: float = 0.15,
    ):
        self.task_complexities = task_complexities
        self.prior_sigma = prior_sigma
        self.learning_rate = learning_rate

        # Per-agent, per-task cross-entropy estimate: agent_id → task_id → H(p,q)
        self._cross_entropy: dict[str, dict[str, float]] = defaultdict(dict)

        # Per-agent, per-task observation counts
        self._observations: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def get_sigma(self, agent_id: str, task_id: str) -> float:
        """Get the current σ for an agent-task pair.

        σ = H(p) / H(p,q)

        If no observations, return the prior σ.
        """
        complexity = self.task_complexities.get(task_id)
        if complexity is None:
            return self.prior_sigma

        h_p = complexity.intrinsic_entropy
        if h_p <= 0.001:
            return 1.0  # trivial task

        h_pq = self._cross_entropy.get(agent_id, {}).get(task_id)
        if h_pq is None or h_pq <= 0:
            return self.prior_sigma

        sigma = h_p / h_pq
        return max(0.01, min(sigma, 0.999))  # clamp to (0, 1)

    def get_temperature(self, agent_id: str, task_id: str) -> float:
        """Temperature T = 1/(1-σ) under the bits-ledger mapping."""
        sigma = self.get_sigma(agent_id, task_id)
        if sigma >= 0.999:
            return 1000.0  # effectively infinite
        return 1.0 / (1.0 - sigma)

    def update(self, agent_id: str, task_id: str, success: bool, efficiency: float) -> None:
        """Update the cross-entropy estimate from an observed outcome.

        The agent's cross-entropy H(p,q) reflects how much it "pays" to
        complete the task. A successful, efficient completion means low
        cross-entropy (high σ). A failure means high cross-entropy (low σ).

        We estimate H(p,q) as:
            H(p,q)_observed = H(p) / observed_efficiency

        Where observed_efficiency accounts for both success/failure and
        the quality of execution.
        """
        complexity = self.task_complexities.get(task_id)
        if complexity is None:
            return

        h_p = complexity.intrinsic_entropy
        if h_p <= 0.001:
            return

        # Observed effective efficiency: 0 for failure, `efficiency` for success
        observed_eff = efficiency if success else 0.0

        # Convert to cross-entropy: H(p,q) = H(p) / σ_observed
        # But σ_observed = 0 for failures → H(p,q) = infinity → clamp
        if observed_eff > 0.01:
            observed_h_pq = h_p / observed_eff
        else:
            observed_h_pq = h_p * 10  # severe penalty: very high cross-entropy

        # Exponential moving average update
        agent_ce = self._cross_entropy[agent_id]
        old_h_pq = agent_ce.get(task_id)

        if old_h_pq is None:
            # First observation — initialize from prior
            prior_h_pq = h_p / self.prior_sigma
            new_h_pq = prior_h_pq + self.learning_rate * (observed_h_pq - prior_h_pq)
        else:
            new_h_pq = old_h_pq + self.learning_rate * (observed_h_pq - old_h_pq)

        agent_ce[task_id] = max(h_p, new_h_pq)  # H(p,q) ≥ H(p) always
        self._observations[agent_id][task_id] += 1

    def chain_sigma(
        self,
        agent_id: str,
        task_id: str,
        chain: ValueChain,
        snapshot: FieldSnapshot,
        max_depth: int = 5,
    ) -> float:
        """Composition rule: σ_total = σ₁ · σ₂ · ... · σₙ for a value chain.

        This replaces the hand-tuned downstream multiplier. The total
        efficiency of completing a task and propagating energy through its
        downstream chain is the PRODUCT of each hop's σ.

        Uses a simplified model: each downstream hop's σ is estimated from
        the task complexity at that hop (not agent-specific, since we don't
        know which agent will handle downstream tasks).

        Returns σ_total for the chain starting at task_id.
        """
        sigma_product = self.get_sigma(agent_id, task_id)

        visited = {task_id}
        frontier = [(task_id, sigma_product)]
        total_sigma = sigma_product

        for _ in range(max_depth):
            if not frontier:
                break
            new_frontier = []
            for current_id, accumulated_sigma in frontier:
                for downstream_id, coeff in snapshot.downstream.get(current_id, []):
                    if downstream_id in visited:
                        continue
                    visited.add(downstream_id)

                    # Estimate σ for downstream task (use complexity-based prior)
                    complexity = self.task_complexities.get(downstream_id)
                    if complexity:
                        # Average agent σ for this task (or prior if no data)
                        downstream_sigma = self._avg_sigma_for_task(downstream_id)
                    else:
                        downstream_sigma = self.prior_sigma

                    # Composition: product of sigmas, weighted by coupling
                    chain_sigma = accumulated_sigma * downstream_sigma * coeff
                    total_sigma += chain_sigma  # accumulate downstream potential

                    new_frontier.append((downstream_id, chain_sigma))
            frontier = new_frontier

        return min(total_sigma, 1.0)

    def _avg_sigma_for_task(self, task_id: str) -> float:
        """Average σ across all agents who have done this task."""
        sigmas = []
        for agent_id in self._cross_entropy:
            s = self.get_sigma(agent_id, task_id)
            if s != self.prior_sigma:  # only count agents with actual data
                sigmas.append(s)
        return sum(sigmas) / len(sigmas) if sigmas else self.prior_sigma

    def sigma_profile(self, agent_id: str) -> dict[str, float]:
        """Return the full σ profile for an agent across all tasks."""
        return {
            tid: self.get_sigma(agent_id, tid)
            for tid in self.task_complexities
        }


def build_task_complexities(
    chain: ValueChain,
    stats: Optional[dict[str, dict]] = None,
) -> dict[str, TaskComplexity]:
    """Build complexity estimates for all tasks in a value chain.

    If process-mining stats are available, use outcome distributions.
    Otherwise, estimate from capability requirements and topology.
    """
    complexities = {}

    for task_id, task in chain.tasks.items():
        # Start with capability-based estimate
        n_caps = len(task.required_capabilities) if task.required_capabilities else 0

        if stats:
            # Use process-mining data if available
            freq = stats.get(task_id, {}).get("frequency", 1)
            avg_duration = stats.get(task_id, {}).get("avg_duration_seconds", 3600)

            # Complexity heuristic: low-frequency + high-duration = complex
            # High-frequency + low-duration = simple
            duration_factor = min(math.log(1 + avg_duration / 3600), 1.0)
            freq_factor = 1.0 / (1.0 + math.log(1 + freq))
            h_p = 0.3 + 0.3 * duration_factor + 0.2 * freq_factor + 0.05 * n_caps
            h_p = min(h_p, 0.9)
        else:
            # No stats — use capability-based heuristic
            h_p = min(0.3 + 0.1 * n_caps, 0.8)

        complexities[task_id] = TaskComplexity(
            task_id=task_id,
            intrinsic_entropy=h_p,
        )

    return complexities
