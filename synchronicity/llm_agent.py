"""
Hermes Agent Adapter — real LLM agent that reads the energy field and reasons
about which task to pick.

This adapter connects Synchronicity's energy field to any LLM (Hermes Agent,
raw OpenAI, local model). The LLM receives a structured field snapshot as
context, reasons about it in natural language, and returns a task selection.

The key insight: an LLM agent can reason about strategic factors that are
impossible to encode in a scoring formula:

  - "Task A has high energy, but 3 other laborers are competing for it.
     Task B has less energy but I'm the only one who can do it, and it
     feeds two downstream chains."

  - "The warehouse is near capacity. If I complete collection now, the
     energy will overflow. I should wait for the manager to clear the
     warehouse first."

  - "I've done sorting 5 times successfully. My model says I'm reliable
     here. The system needs sorting done — I should prioritize it."

This is the actual intelligence layer that makes Synchronicity more than
a scoring algorithm.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from typing import Optional

from synchronicity.field import FieldSnapshot
from synchronicity.value_chain import ValueChain
from synchronicity.agents import (
    AgentCapability, AgentDecision, BaseAgent, _softmax_sample,
)

logger = logging.getLogger(__name__)


class LLMAgent(BaseAgent):
    """Agent that uses a real LLM to decide which task to pick.

    The agent formats the field snapshot as a natural language prompt,
    sends it to an LLM, and parses the response into a task selection.

    Supports two backends:
      1. Hermes CLI: calls `hermes --message "..."` via subprocess
      2. Direct API: calls an OpenAI-compatible endpoint

    Falls back to FieldAgent logic if the LLM is unavailable or returns
    an unparseable response.
    """

    def __init__(
        self,
        capability: AgentCapability,
        chain: ValueChain,
        backend: str = "hermes",
        model: Optional[str] = None,
        fallback_temperature: float = 1.0,
    ):
        super().__init__(capability)
        self.chain = chain
        self.backend = backend
        self.model = model
        self.fallback_temperature = fallback_temperature

        # Competition learning (same as FieldAgent)
        self._competition_pressure: dict[str, float] = {}
        self._last_energy_observed: dict[str, float] = {}
        self._llm_calls: int = 0
        self._llm_failures: int = 0

    def observe(self, snapshot: FieldSnapshot) -> None:
        """Update competition model from observed energy changes."""
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

    def _format_snapshot_prompt(self, snap: FieldSnapshot) -> str:
        """Format the field snapshot as a natural language context for the LLM."""
        lines = [
            "You are an agent in a decentralized economic coordination system.",
            "You must choose ONE task to work on this turn.",
            "",
            "Available tasks (sorted by energy):",
        ]

        sorted_tasks = sorted(snap.task_energy.items(), key=lambda x: -x[1])
        for task_id, energy in sorted_tasks:
            if energy < 0.5:
                continue

            task_caps = snap.task_capabilities.get(task_id, frozenset())
            if task_caps and not task_caps.intersection(self.capability.capabilities):
                continue

            downstream = snap.downstream.get(task_id, [])
            pressure = self._competition_pressure.get(task_id, 0.0)
            task_type = snap.task_types.get(task_id, "unknown")

            lines.append(f"\n  TASK: {task_id}")
            lines.append(f"    energy: {energy:.1f}")
            lines.append(f"    type: {task_type}")
            lines.append(f"    downstream chains: {len(downstream)} tasks")
            if pressure > 0.1:
                lines.append(f"    competition: HIGH (pressure={pressure:.2f})")
            else:
                lines.append(f"    competition: low")

        lines.extend([
            "",
            f"Your capabilities: {', '.join(sorted(self.capability.capabilities))}",
            f"Your efficiency: {self.capability.base_efficiency:.0%}",
            "",
            "STRATEGY: Pick the task that maximizes total system value, not just",
            "your individual reward. Consider: downstream value chains, competition",
            "from other agents, and your unique capability fit.",
            "",
            "Respond with ONLY the task name (e.g., 'sort_center'). Nothing else.",
        ])

        return "\n".join(lines)

    def _call_hermes(self, prompt: str) -> Optional[str]:
        """Call Hermes Agent CLI to get an LLM decision."""
        try:
            cmd = [
                "hermes", "-z", prompt,
                "--ignore-rules", "--ignore-user-config",
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
            )
            self._llm_calls += 1
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            else:
                logger.debug(f"Hermes CLI returned {result.returncode}: {result.stderr[:200]}")
                self._llm_failures += 1
                return None
        except Exception as e:
            logger.debug(f"Hermes CLI call failed: {e}")
            self._llm_failures += 1
            return None

    def _call_api(self, prompt: str) -> Optional[str]:
        """Call an OpenAI-compatible API endpoint."""
        try:
            import openai
            base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
            api_key = os.environ.get("OPENAI_API_KEY", "ollama")
            model = self.model or os.environ.get("SYNCHRONICITY_LLM_MODEL", "qwen2.5:7b")

            client = openai.OpenAI(base_url=base_url, api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.7,
            )
            self._llm_calls += 1
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.debug(f"LLM API call failed: {e}")
            self._llm_failures += 1
            return None

    def _parse_task_choice(self, response: str, snap: FieldSnapshot) -> Optional[str]:
        """Extract a task ID from the LLM's text response."""
        if not response:
            return None

        response_lower = response.lower().strip().strip(".'\"")

        # Direct match
        for task_id in snap.task_energy:
            if response_lower == task_id:
                return task_id

        # Substring match
        for task_id in snap.task_energy:
            if task_id in response_lower:
                return task_id

        # Try normalizing (remove spaces, underscores)
        response_clean = response_lower.replace(" ", "_").replace("-", "_")
        for task_id in snap.task_energy:
            if task_id in response_clean:
                return task_id

        return None

    def _fallback_decide(self) -> AgentDecision:
        """Fallback when LLM is unavailable: use a simple heuristic."""
        snap = self._last_snapshot
        if snap is None:
            return AgentDecision(task_id=None, reasoning="no snapshot")

        scores: dict[str, float] = {}
        for task_id, energy in snap.task_energy.items():
            if energy <= 0.1:
                continue
            task_caps = snap.task_capabilities.get(task_id, frozenset())
            if task_caps and not task_caps.intersection(self.capability.capabilities):
                continue
            pressure = self._competition_pressure.get(task_id, 0.0)
            scores[task_id] = energy / (1.0 + pressure)

        if not scores:
            return AgentDecision(task_id=None, reasoning="no viable tasks")

        best_task = _softmax_sample(scores, temperature=self.fallback_temperature)
        return AgentDecision(
            task_id=best_task,
            expected_efficiency=self.capability.base_efficiency,
            reasoning="llm-fallback: energy/competition heuristic",
        )

    def decide(self) -> AgentDecision:
        snap = self._last_snapshot
        if snap is None:
            return AgentDecision(task_id=None, reasoning="no snapshot")

        # Filter viable tasks
        viable = {}
        for task_id, energy in snap.task_energy.items():
            if energy > 0.5:
                task_caps = snap.task_capabilities.get(task_id, frozenset())
                if not task_caps or task_caps.intersection(self.capability.capabilities):
                    viable[task_id] = energy

        if not viable:
            return AgentDecision(task_id=None, reasoning="no viable tasks")

        # Try LLM
        prompt = self._format_snapshot_prompt(snap)

        if self.backend == "hermes":
            response = self._call_hermes(prompt)
        else:
            response = self._call_api(prompt)

        if response:
            task_choice = self._parse_task_choice(response, snap)
            if task_choice and task_choice in viable:
                return AgentDecision(
                    task_id=task_choice,
                    expected_efficiency=self.capability.base_efficiency,
                    reasoning=f"llm: {response[:80]}",
                )

        # Fallback
        return self._fallback_decide()

    def stats(self) -> dict:
        s = super().stats()
        s["llm_calls"] = self._llm_calls
        s["llm_failures"] = self._llm_failures
        return s
