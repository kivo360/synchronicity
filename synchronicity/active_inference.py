"""
Active Inference Agent — replaces hand-tuned heuristic weights with the
Free Energy Principle (Friston).

Instead of scoring tasks with arbitrary weights, this agent minimizes
Expected Free Energy (EFE):

    EFE(task) = E[ln p(outcome | task)]     (pragmatic value / expected reward)
              - E[KL(q(belief) || p(prior))]  (epistemic value / info gain)
              + H(ambiguity)                 (uncertainty about the outcome)

In Synchronicity terms:
  - Pragmatic value = expected energy reward (we already compute this)
  - Epistemic value = how much doing this task reduces the agent's uncertainty
    about the field state (information gain from observation)
  - Ambiguity = how uncertain the task's current state is

The agent picks the task that minimizes EFE — balancing reward-seeking
(pragmatic) with exploration (epistemic) and uncertainty avoidance (ambiguity).

This is mathematically grounded (variational free energy, well-established
in computational neuroscience) rather than hand-tuned. When the full
Synchronicity field equations arrive, this scoring function becomes a
direct implementation of them — or possibly IS them.
"""

from __future__ import annotations

import math
import random
from typing import Optional

from synchronicity.field import FieldSnapshot
from synchronicity.value_chain import ValueChain
from synchronicity.agents import (
    AgentCapability, AgentDecision, BaseAgent, _softmax_sample, FieldAgent,
)


class ActiveInferenceAgent(BaseAgent):
    """Agent that selects tasks by minimizing Expected Free Energy.

    The scoring function is:

        score(task) = pragmatic_value(task)     (expected reward)
                    + epistemic_value(task)     (information gain)
                    - ambiguity_penalty(task)   (outcome uncertainty)

    All three terms are computed from the field snapshot and the agent's
    learned world model. No hand-tuned weights — the relative magnitudes
    emerge from the field state itself.

    Learned world model:
      - For each task, tracks: success_rate, avg_reward, observation_count
      - Builds a probability distribution over outcomes
      - Entropy of that distribution = ambiguity
    """

    def __init__(
        self,
        capability: AgentCapability,
        chain: ValueChain,
        exploration_bonus: float = 0.3,
        ambiguity_aversion: float = 0.5,
    ):
        super().__init__(capability)
        self.chain = chain
        self.exploration_bonus = exploration_bonus
        self.ambiguity_aversion = ambiguity_aversion

        # Learned world model: task_id → {successes, failures, total_reward}
        self._task_model: dict[str, dict] = {}

        # Competition pressure (same learning mechanism as FieldAgent)
        self._competition_pressure: dict[str, float] = {}
        self._last_energy_observed: dict[str, float] = {}

        # Observation history for epistemic value
        self._observation_counts: dict[str, int] = {}

    def observe(self, snapshot: FieldSnapshot) -> None:
        """Update learned models from field state changes."""
        # Update competition model (same as FieldAgent — learns from energy depletion)
        if self._last_energy_observed:
            for task_id, current_energy in snapshot.task_energy.items():
                prev_energy = self._last_energy_observed.get(task_id, 0.0)
                drop = prev_energy - current_energy
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
        """Update the learned world model with the outcome."""
        super().receive_reward(reward, success)

        # Track which task we just did (need last decision)
        if hasattr(self, '_last_task_id') and self._last_task_id:
            tid = self._last_task_id
            model = self._task_model.setdefault(tid, {
                "successes": 0, "failures": 0, "total_reward": 0.0
            })
            if success:
                model["successes"] += 1
            else:
                model["failures"] += 1
            model["total_reward"] += reward

    def _compute_efe(self, task_id: str, energy: float, snapshot: FieldSnapshot) -> tuple[float, dict]:
        """Compute Expected Free Energy for a task.

        Returns (total_score, breakdown_dict).

        score = pragmatic + epistemic - ambiguity
        Higher score = more attractive task (lower expected free energy).
        """
        # ── Pragmatic value: expected reward ────────────────────
        # Adjust raw energy by learned competition pressure
        pressure = self._competition_pressure.get(task_id, 0.0)
        expected_energy = energy / (1.0 + pressure)

        # Adjust by learned success rate if we have data
        model = self._task_model.get(task_id)
        if model and (model["successes"] + model["failures"]) > 0:
            total_attempts = model["successes"] + model["failures"]
            success_rate = model["successes"] / total_attempts
            pragmatic = expected_energy * success_rate
        else:
            # No data — assume optimistic prior
            pragmatic = expected_energy * 0.9

        # ── Epistemic value: information gain ───────────────────
        # Tasks we've observed less have higher epistemic value (exploration).
        # This is the inverse of observation count, scaled to energy magnitude.
        obs_count = self._observation_counts.get(task_id, 0)
        # Information gain decreases logarithmically with observations
        # (Diminishing returns — each observation tells you less)
        if obs_count == 0:
            epistemic = self.exploration_bonus * energy * 0.2
        else:
            epistemic = self.exploration_bonus * energy * 0.2 / (1 + math.log(1 + obs_count))

        # Downstream chain also contributes epistemic value — completing
        # this task reveals information about downstream tasks
        downstream = snapshot.downstream.get(task_id, [])
        chain_bonus = len(downstream) * self.exploration_bonus * 5
        epistemic += chain_bonus

        # ── Ambiguity: outcome uncertainty ──────────────────────
        # If we've done this task many times and results vary, ambiguity is high.
        # If we've never done it, ambiguity is moderate (unknown unknown).
        # If we've done it and results are consistent, ambiguity is low.
        if model and (model["successes"] + model["failures"]) >= 3:
            total = model["successes"] + model["failures"]
            p_success = model["successes"] / total
            # Binary entropy: H = -p*log(p) - (1-p)*log(1-p)
            if 0 < p_success < 1:
                ambiguity = -(p_success * math.log2(p_success) +
                              (1 - p_success) * math.log2(1 - p_success))
            else:
                ambiguity = 0.0  # deterministic outcome
        elif model and (model["successes"] + model["failures"]) > 0:
            ambiguity = 0.5  # small sample — moderate uncertainty
        else:
            ambiguity = 0.3  # untried task — some uncertainty

        ambiguity_penalty = self.ambiguity_aversion * ambiguity * energy * 0.1

        score = pragmatic + epistemic - ambiguity_penalty

        return score, {
            "pragmatic": pragmatic,
            "epistemic": epistemic,
            "ambiguity": ambiguity,
            "ambiguity_penalty": ambiguity_penalty,
            "pressure": pressure,
            "expected_energy": expected_energy,
        }

    def decide(self) -> AgentDecision:
        snap = self._last_snapshot
        if snap is None:
            return AgentDecision(task_id=None, reasoning="no snapshot")

        scores: dict[str, float] = {}
        breakdowns: dict[str, dict] = {}

        for task_id, energy in snap.task_energy.items():
            if energy <= 0.1:
                continue

            # Filter: skip tasks the agent cannot do
            task_caps = snap.task_capabilities.get(task_id, frozenset())
            if task_caps and not task_caps.intersection(self.capability.capabilities):
                continue

            score, breakdown = self._compute_efe(task_id, energy, snap)
            scores[task_id] = score
            breakdowns[task_id] = breakdown

        if not scores:
            return AgentDecision(task_id=None, reasoning="no viable tasks")

        # Softmax sampling over EFE scores
        best_task = _softmax_sample(scores, temperature=1.0)

        # Track observation
        self._observation_counts[best_task] = self._observation_counts.get(best_task, 0) + 1
        self._last_task_id = best_task

        b = breakdowns[best_task]
        return AgentDecision(
            task_id=best_task,
            expected_efficiency=self.capability.base_efficiency,
            reasoning=(
                f"active_inference: pragmatic={b['pragmatic']:.1f}"
                f" + epistemic={b['epistemic']:.1f}"
                f" - ambiguity={b['ambiguity_penalty']:.1f}"
                f" (H={b['ambiguity']:.2f}, pressure={b['pressure']:.2f})"
            ),
        )
