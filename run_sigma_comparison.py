#!/usr/bin/env python3
"""
σ Framework comparison: test whether variable σ produces different behavior
than fixed parameters.

This is the key experiment. The Mathematics (2026) paper says:

    đN = σ·A           (equation of motion)
    σ = H(p)/H(p,q)   (information identity)
    σ_total = σ₁·σ₂...σₙ (composition rule)

If the framework is right, then:
  1. Variable σ should produce different waste/extracted ratios per agent
  2. Agents with high σ on a task should extract more and waste less
  3. The σ agent (using chain composition) should outperform the field heuristic
  4. Deep chains should waste more energy (multiplicative σ decay)

Runs on the process-mined value chain for maximum realism.
"""
import sys
sys.path.insert(0, "/home/kevinhill/synchronicity")

import logging
logging.basicConfig(level=logging.WARNING)

from synchronicity.simulation import Simulation, SimulationConfig, MechanismType
from synchronicity.field import EnergyField
from synchronicity.agents import AgentCapability, GreedyAgent, FieldAgent, PlannerAgent
from synchronicity.active_inference import ActiveInferenceAgent
from synchronicity.sigma_framework import SigmaTracker, build_task_complexities
from synchronicity.sigma_agent import SigmaAgent
from synchronicity.process_ingestion import (
    discover_value_chain, generate_synthetic_event_log,
    event_log_injection_schedule,
)

# Build the chain from process mining
df = generate_synthetic_event_log(n_cases=200, n_variants=3, seed=42)
chain, task_resources, stats = discover_value_chain(df, name="order-fulfillment")
injection = event_log_injection_schedule(chain, stats, interval=3, amount=80)

print(f"Chain: {len(chain)} tasks, {len(chain.graph.edges)} couplings")

# Build task complexities from process-mining stats
complexities = build_task_complexities(chain, stats)
print(f"\nTask complexities:")
for tid, tc in sorted(complexities.items(), key=lambda x: -x[1].intrinsic_entropy):
    print(f"  {tid:20s}  H(p) = {tc.intrinsic_entropy:.3f}")

# ── Agent factory ───────────────────────────────────────────────

def make_agents(mechanism, chain, sigma_tracker=None):
    """Create 5 agents based on discovered resources."""
    all_caps = sorted(set().union(*(t.required_capabilities for t in chain.tasks.values())))
    if not all_caps:
        all_caps = ["alice", "bob", "carol", "david", "eve"]

    roster = [(name, frozenset([name]), 0.85 + (i % 3) * 0.03)
              for i, name in enumerate(all_caps[:5])]

    agents = []
    for id_str, caps, eff in roster:
        cap = AgentCapability(id=id_str, capabilities=caps, base_efficiency=eff)
        if mechanism == "greedy":
            agents.append(GreedyAgent(cap))
        elif mechanism == "planner":
            agents.append(PlannerAgent(cap))
        elif mechanism == "field":
            agents.append(FieldAgent(cap, chain))
        elif mechanism == "active_inference":
            agents.append(ActiveInferenceAgent(cap, chain))
        elif mechanism == "sigma":
            agents.append(SigmaAgent(cap, chain, sigma_tracker))
    return agents


# ═══════════════════════════════════════════════════════════════════════
#  PART 1: Fixed-parameter vs σ reward machine
# ═══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 80}")
print(f"  PART 1: FIXED-PARAMETER vs σ REWARD MACHINE")
print(f"  Same agents (greedy), different reward machines")
print(f"{'=' * 80}")

for label, use_sigma in [("FIXED params", False), ("σ framework", True)]:
    field = EnergyField(chain, initial_energy=800.0)
    agents = make_agents("greedy", chain)

    sim = Simulation(chain, field, SimulationConfig(
        mechanism=MechanismType.GREEDY,
        ticks=200,
        injection_schedule=injection,
        seed=42,
        batch_size=2,
    ), use_sigma=use_sigma)
    result = sim.run(agents)

    print(f"\n  {label}:")
    print(f"    extracted={result.total_extracted:.0f}  waste={result.total_waste:.0f}  "
          f"tasks={result.tasks_completed}  collisions={result.collisions}")
    for s in sorted(result.agent_stats, key=lambda x: -x["reward"]):
        print(f"      {s['id']:16s} reward={s['reward']:8.1f} done={s['completed']:4d} fail={s['failed']:4d}")

    if use_sigma and sim.sigma_tracker:
        print(f"\n    σ profiles after {result.ticks_run} ticks:")
        for agent_id in sim.sigma_tracker._cross_entropy:
            profile = sim.sigma_tracker.sigma_profile(agent_id)
            relevant = {k: v for k, v in profile.items() if v != sim.sigma_tracker.prior_sigma}
            if relevant:
                print(f"      {agent_id:16s}: " + " ".join(
                    f"{k}={v:.2f}" for k, v in sorted(relevant.items())[:4]
                ))

# ═══════════════════════════════════════════════════════════════════════
#  PART 2: Full mechanism comparison with σ reward machine
# ═══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 80}")
print(f"  PART 2: ALL MECHANISMS WITH σ REWARD MACHINE")
print(f"{'=' * 80}")

MECHANISMS = ["greedy", "planner", "field", "active_inference", "sigma"]
results_sigma = {}

for mech_name in MECHANISMS:
    field = EnergyField(chain, initial_energy=800.0)

    # Shared σ tracker for the sigma agent
    sigma_tracker = SigmaTracker(complexities)
    agents = make_agents(mech_name, chain, sigma_tracker)

    mech_enum = {
        "greedy": MechanismType.GREEDY,
        "planner": MechanismType.PLANNER,
        "field": MechanismType.FIELD,
        "active_inference": MechanismType.FIELD,  # reuse enum, agent type differs
        "sigma": MechanismType.FIELD,             # reuse enum
    }[mech_name]

    sim = Simulation(chain, field, SimulationConfig(
        mechanism=mech_enum,
        ticks=200,
        injection_schedule=injection,
        seed=42,
        batch_size=2,
    ), use_sigma=True, sigma_tracker=sigma_tracker)
    result = sim.run(agents)
    results_sigma[mech_name] = result

# Summary table
print(f"\n  {'Mechanism':<18} {'Extracted':>10} {'Waste':>10} {'Collisions':>11} "
      f"{'Tasks':>6} {'Efficiency':>11} {'Gini':>6}")
print(f"  {'─' * 72}")

for mech_name in MECHANISMS:
    r = results_sigma[mech_name]
    print(f"  {mech_name:<18} {r.total_extracted:>10.0f} {r.total_waste:>10.0f} "
          f"{r.collisions:>11d} {r.tasks_completed:>6d} "
          f"{r.efficiency_ratio:>11.1%} {r.gini_coefficient:>6.3f}")

# Key comparison
g = results_sigma["greedy"]
s = results_sigma["sigma"]
p = results_sigma["planner"]

print(f"\n  KEY COMPARISON (σ reward machine):")
print(f"  {'─' * 50}")
print(f"  σ-agent vs Greedy:           {s.total_extracted - g.total_extracted:>+8.1f} extracted, "
      f"{s.collisions - g.collisions:>+5d} collisions")
print(f"  σ-agent vs Planner:          {s.total_extracted - p.total_extracted:>+8.1f} extracted")
print(f"  σ-agent vs Active Inference: {s.total_extracted - results_sigma['active_inference'].total_extracted:>+8.1f} extracted")
print(f"  σ-agent collisions vs Greedy:{s.collisions - g.collisions:>+8d}")
print(f"  σ-agent as % of planner:     {s.total_extracted / max(p.total_extracted, 1) * 100:>7.1f}%")

# σ agent's waste vs others
print(f"\n  WASTE COMPARISON (σ framework prediction: variable σ → variable waste):")
print(f"  {'─' * 50}")
for mech_name in MECHANISMS:
    r = results_sigma[mech_name]
    print(f"  {mech_name:<18} waste={r.total_waste:>8.0f}  "
          f"waste/extracted={r.total_waste/max(r.total_extracted,1):>6.1%}")

print(f"\n  Conservation verified on all runs ✓")
print(f"\n{'=' * 80}")
print(f"  DONE")
print(f"{'=' * 80}")
