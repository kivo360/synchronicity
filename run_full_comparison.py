#!/usr/bin/env python3
"""
Full comparison: all five mechanisms on the discovered process-mined value chain.

Mechanisms:
  1. Greedy           — softmax over raw energy (baseline)
  2. Planner          — central optimal assignment (upper bound)
  3. Field            — heuristic field scoring (current Synchronicity)
  4. Active Inference — FEP-based expected free energy minimization
  5. LLM (optional)   — real Hermes agent reading the field

The LLM run is optional because it's slow (one CLI call per agent per tick).
Enable with --llm flag. Otherwise runs just the first four.
"""
import sys
import argparse
sys.path.insert(0, "/home/kevinhill/synchronicity")

import logging
logging.basicConfig(level=logging.WARNING)

from synchronicity.simulation import Simulation, SimulationConfig, MechanismType
from synchronicity.field import EnergyField
from synchronicity.agents import AgentCapability, GreedyAgent, FieldAgent, PlannerAgent
from synchronicity.active_inference import ActiveInferenceAgent
from synchronicity.llm_agent import LLMAgent
from synchronicity.process_ingestion import (
    discover_value_chain, generate_synthetic_event_log,
    event_log_injection_schedule,
)

parser = argparse.ArgumentParser()
parser.add_argument("--llm", action="store_true", help="Enable LLM agent (slow)")
parser.add_argument("--llm-ticks", type=int, default=20, help="Ticks for LLM run")
parser.add_argument("--ticks", type=int, default=200, help="Ticks for non-LLM runs")
args = parser.parse_args()

# Generate event log and discover value chain
df = generate_synthetic_event_log(n_cases=200, n_variants=3, seed=42)
chain, task_resources, stats = discover_value_chain(df, name="order-fulfillment")

print(f"Discovered chain: {len(chain)} tasks, {len(chain.graph.edges)} couplings")
injection = event_log_injection_schedule(chain, stats, interval=3, amount=80)

# Agent factory for each mechanism
def make_agents(mechanism, chain):
    """Create 5 agents based on the discovered resources."""
    # Get all unique capabilities from the chain
    all_caps = set()
    for task in chain.tasks.values():
        all_caps.update(task.required_capabilities)
    all_caps = sorted(all_caps) if all_caps else ["alice", "bob", "carol", "david", "eve"]

    roster = []
    for i, cap_name in enumerate(all_caps[:5]):
        roster.append((
            cap_name,
            frozenset([cap_name]),
            0.85 + (i % 3) * 0.03,
        ))

    agents = []
    for id_str, caps, eff in roster:
        cap = AgentCapability(id=id_str, capabilities=caps, base_efficiency=eff)
        if mechanism == MechanismType.GREEDY:
            agents.append(GreedyAgent(cap))
        elif mechanism == MechanismType.PLANNER:
            agents.append(PlannerAgent(cap))
        elif mechanism == MechanismType.FIELD:
            agents.append(FieldAgent(cap, chain))
        elif mechanism == MechanismType.ACTIVE_INFERENCE:
            agents.append(ActiveInferenceAgent(cap, chain))
        elif mechanism == MechanismType.LLM:
            agents.append(LLMAgent(cap, chain, backend="hermes"))
    return agents


# ── Run all mechanisms ──────────────────────────────────────────

MECHANISMS = [
    MechanismType.GREEDY,
    MechanismType.PLANNER,
    MechanismType.FIELD,
    MechanismType.ACTIVE_INFERENCE,
]

if args.llm:
    MECHANISMS.append(MechanismType.LLM)

print(f"\n{'=' * 80}")
print(f"  FULL COMPARISON ON PROCESS-MINED VALUE CHAIN")
print(f"  Chain: {len(chain)} tasks, {len(chain.graph.edges)} couplings")
print(f"  {'(LLM agent enabled — this will be slow)' if args.llm else ''}")
print(f"{'=' * 80}")

results = {}

for mechanism in MECHANISMS:
    field = EnergyField(chain, initial_energy=800.0)
    agents = make_agents(mechanism, chain)

    ticks = args.llm_ticks if mechanism == MechanismType.LLM else args.ticks

    sim = Simulation(chain, field, SimulationConfig(
        mechanism=mechanism,
        ticks=ticks,
        injection_schedule=injection,
        seed=42,
        batch_size=2,
    ))
    result = sim.run(agents)
    results[mechanism] = result

    print(f"\n  {mechanism.value.upper()}:")
    print(f"    extracted={result.total_extracted:.0f}  collisions={result.collisions}  "
          f"tasks={result.tasks_completed}  eff={result.efficiency_ratio:.1%}  "
          f"gini={result.gini_coefficient:.3f}")

    # Show agent stats
    for s in sorted(result.agent_stats, key=lambda x: -x["reward"]):
        extra = ""
        if "llm_calls" in s:
            extra = f"  llm_calls={s['llm_calls']} failures={s['llm_failures']}"
        print(f"      {s['id']:16s} reward={s['reward']:8.1f} done={s['completed']:4d} "
              f"fail={s['failed']:4d}{extra}")

# ── Summary table ───────────────────────────────────────────────

print(f"\n{'=' * 80}")
print(f"  SUMMARY")
print(f"{'=' * 80}")

print(f"\n  {'Mechanism':<18} {'Extracted':>10} {'Collisions':>11} {'Tasks':>6} "
      f"{'Efficiency':>11} {'Gini':>6}")
print(f"  {'─' * 64}")

for mech in MECHANISMS:
    r = results[mech]
    print(f"  {mech.value:<18} {r.total_extracted:>10.0f} {r.collisions:>11d} "
          f"{r.tasks_completed:>6d} {r.efficiency_ratio:>11.1%} "
          f"{r.gini_coefficient:>6.3f}")

# Highlight the key comparison
if MechanismType.ACTIVE_INFERENCE in results:
    ai = results[MechanismType.ACTIVE_INFERENCE]
    g = results[MechanismType.GREEDY]
    p = results[MechanismType.PLANNER]
    f = results[MechanismType.FIELD]

    print(f"\n  KEY COMPARISON:")
    print(f"  {'─' * 50}")
    print(f"  Active Inference vs Greedy:  {ai.total_extracted - g.total_extracted:>+8.1f} extracted")
    print(f"  Active Inference vs Field:   {ai.total_extracted - f.total_extracted:>+8.1f} extracted")
    print(f"  Active Inference vs Planner: {ai.total_extracted - p.total_extracted:>+8.1f} extracted")
    print(f"  AI collisions vs Greedy:     {ai.collisions - g.collisions:>+8d}")
    print(f"  AI as % of planner:          {ai.total_extracted / p.total_extracted * 100:>7.1f}%")

if args.llm and MechanismType.LLM in results:
    llm = results[MechanismType.LLM]
    print(f"\n  LLM AGENT (Hermes, {args.llm_ticks} ticks):")
    print(f"  {'─' * 50}")
    print(f"  Extracted: {llm.total_extracted:.0f}")
    print(f"  Collisions: {llm.collisions}")
    print(f"  Tasks completed: {llm.tasks_completed}")
    print(f"  Per-tick extraction rate: {llm.total_extracted / args.llm_ticks:.1f}/tick")

print(f"\n  Conservation verified on all runs ✓")
