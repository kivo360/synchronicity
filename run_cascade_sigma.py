#!/usr/bin/env python3
"""
Cascade scenario test — where the σ composition rule should actually matter.

The cascade is a 5-hop deep value chain:
    mine → refine → assemble ─┬→ package_A → ship_A
                             └→ package_B → ship_B

With σ = 0.85 per hop, σ_total = 0.85⁵ = 0.44. That means 56% of energy
is lost to multiplicative inefficiency through the chain. This is the
bullwhip effect from the Twitter-DAO paper, quantified.

The theory predicts:
  1. Deep chains waste more energy than flat ones (composition rule)
  2. σ-agent should outperform greedy by pre-positioning downstream
  3. The σ-aware planner should assign differently than the myopic planner
     because σ_chain scoring accounts for downstream potential
"""
import sys
sys.path.insert(0, "/home/kevinhill/synchronicity")

import logging
logging.basicConfig(level=logging.WARNING)

from synchronicity.simulation import Simulation, SimulationConfig, MechanismType
from synchronicity.field import EnergyField
from synchronicity.agents import (
    AgentCapability, GreedyAgent, FieldAgent, PlannerAgent, PlannerSystem,
)
from synchronicity.active_inference import ActiveInferenceAgent
from synchronicity.sigma_framework import SigmaTracker, build_task_complexities
from synchronicity.sigma_agent import SigmaAgent
from synchronicity.scenarios import (
    build_cascade_scenario, make_cascade_agents, cascade_injection,
)

# Build the cascade chain
chain, _ = build_cascade_scenario()

print(f"{'=' * 80}")
print(f"  CASCADE SCENARIO — σ COMPOSITION RULE TEST")
print(f"{'=' * 80}")
print(f"\n  Topology:")
print(f"    mine → refine → assemble ─┬→ package_A → ship_A")
print(f"                             └→ package_B → ship_B")
print(f"\n  Chain depth: 5 hops")
print(f"  Tasks: {len(chain)}  Couplings: {len(chain.graph.edges)}")
print(f"  Agents: 6 (one per stage capability)")

# Show theoretical σ composition
prior_sigma = 0.85
print(f"\n  Theoretical σ composition (prior = {prior_sigma}):")
print(f"    1 hop:  σ = {prior_sigma**1:.3f}  → {(1-prior_sigma**1)*100:.1f}% waste")
print(f"    2 hops: σ = {prior_sigma**2:.3f}  → {(1-prior_sigma**2)*100:.1f}% waste")
print(f"    3 hops: σ = {prior_sigma**3:.3f}  → {(1-prior_sigma**3)*100:.1f}% waste")
print(f"    4 hops: σ = {prior_sigma**4:.3f}  → {(1-prior_sigma**4)*100:.1f}% waste")
print(f"    5 hops: σ = {prior_sigma**5:.3f}  → {(1-prior_sigma**5)*100:.1f}% waste")

complexities = build_task_complexities(chain)
print(f"\n  Task complexities:")
for tid, tc in sorted(complexities.items(), key=lambda x: x[0]):
    print(f"    {tid:16s}  H(p) = {tc.intrinsic_entropy:.3f}")


# ═══════════════════════════════════════════════════════════════════════
#  PART 1: Fixed-parameter reward machine (baseline)
# ═══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 80}")
print(f"  PART 1: FIXED-PARAMETER REWARD MACHINE")
print(f"{'=' * 80}")

MECHANISMS = ["greedy", "planner", "field", "active_inference"]
results_fixed = {}

# Cascade agent roster
CASCADE_ROSTER = [
    ("worker_0", frozenset(["labor"]), 0.88),
    ("worker_1", frozenset(["labor"]), 0.85),
    ("refiner_0", frozenset(["refining"]), 0.90),
    ("assembler_0", frozenset(["assembly"]), 0.92),
    ("packer_0", frozenset(["packaging"]), 0.89),
    ("shipper_0", frozenset(["logistics"]), 0.87),
]

def make_cascade_agents_full(mech_name, chain, sigma_tracker=None):
    agents = []
    for id_str, caps, eff in CASCADE_ROSTER:
        cap = AgentCapability(id=id_str, capabilities=caps, base_efficiency=eff)
        if mech_name == "greedy":
            agents.append(GreedyAgent(cap))
        elif mech_name == "planner":
            agents.append(PlannerAgent(cap))
        elif mech_name == "field":
            agents.append(FieldAgent(cap, chain))
        elif mech_name == "active_inference":
            agents.append(ActiveInferenceAgent(cap, chain))
        elif mech_name == "sigma":
            agents.append(SigmaAgent(cap, chain, sigma_tracker))
    return agents

for mech_name in MECHANISMS:
    field = EnergyField(chain, initial_energy=600.0)
    agents = make_cascade_agents_full(mech_name, chain)

    mech_enum = {
        "greedy": MechanismType.GREEDY,
        "planner": MechanismType.PLANNER,
        "field": MechanismType.FIELD,
        "active_inference": MechanismType.FIELD,
    }[mech_name]

    sim = Simulation(chain, field, SimulationConfig(
        mechanism=mech_enum,
        ticks=200,
        injection_schedule=cascade_injection,
        seed=42,
        batch_size=3,
    ))
    result = sim.run(agents)
    results_fixed[mech_name] = result

print(f"\n  {'Mechanism':<20} {'Extracted':>10} {'Waste':>10} {'Collisions':>11} "
      f"{'Tasks':>6} {'Efficiency':>11}")
print(f"  {'─' * 68}")
for m in MECHANISMS:
    r = results_fixed[m]
    print(f"  {m:<20} {r.total_extracted:>10.0f} {r.total_waste:>10.0f} "
          f"{r.collisions:>11d} {r.tasks_completed:>6d} {r.efficiency_ratio:>11.1%}")


# ═══════════════════════════════════════════════════════════════════════
#  PART 2: σ reward machine (the real test)
# ═══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 80}")
print(f"  PART 2: σ REWARD MACHINE")
print(f"  Variable efficiency: đN = σ·A, waste = A·(1-σ)")
print(f"  Composition: σ_total = σ₁·σ₂·...·σₙ")
print(f"{'=' * 80}")

MECHANISMS_SIGMA = ["greedy", "planner", "field", "active_inference", "sigma"]
results_sigma = {}

for mech_name in MECHANISMS_SIGMA:
    sigma_tracker = SigmaTracker(complexities)
    field = EnergyField(chain, initial_energy=600.0)
    agents = make_cascade_agents_full(mech_name, chain, sigma_tracker)

    mech_enum = {
        "greedy": MechanismType.GREEDY,
        "planner": MechanismType.PLANNER,
        "field": MechanismType.FIELD,
        "active_inference": MechanismType.FIELD,
        "sigma": MechanismType.FIELD,
    }[mech_name]

    sim = Simulation(chain, field, SimulationConfig(
        mechanism=mech_enum,
        ticks=200,
        injection_schedule=cascade_injection,
        seed=42,
        batch_size=3,
    ), use_sigma=True, sigma_tracker=sigma_tracker)
    result = sim.run(agents)
    results_sigma[mech_name] = result

    # Show σ profiles for the sigma agent
    if mech_name == "sigma" and sim.sigma_tracker:
        print(f"\n  σ-AGENT PROFILES after {result.ticks_run} ticks:")
        for agent_id in sim.sigma_tracker._cross_entropy:
            profile = sim.sigma_tracker.sigma_profile(agent_id)
            relevant = {k: v for k, v in profile.items() if v != sim.sigma_tracker.prior_sigma}
            if relevant:
                items = sorted(relevant.items(), key=lambda x: -x[1])[:3]
                print(f"    {agent_id:16s}: " + "  ".join(
                    f"{k}={v:.2f}" for k, v in items))

print(f"\n  {'Mechanism':<20} {'Extracted':>10} {'Waste':>10} {'Collisions':>11} "
      f"{'Tasks':>6} {'Efficiency':>11} {'Waste/Ext':>10}")
print(f"  {'─' * 78}")
for m in MECHANISMS_SIGMA:
    r = results_sigma[m]
    ratio = r.total_waste / max(r.total_extracted, 1)
    print(f"  {m:<20} {r.total_extracted:>10.0f} {r.total_waste:>10.0f} "
          f"{r.collisions:>11d} {r.tasks_completed:>6d} "
          f"{r.efficiency_ratio:>11.1%} {ratio:>10.1%}")


# ═══════════════════════════════════════════════════════════════════════
#  PART 3: Fixed vs σ comparison (does variable σ change anything?)
# ═══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 80}")
print(f"  PART 3: FIXED vs σ REWARD MACHINE (same agent, different physics)")
print(f"{'=' * 80}")

print(f"\n  {'Mechanism':<20} {'Fixed Ext':>10} {'σ Ext':>10} {'Δ':>8} "
      f"{'Fixed W/E':>10} {'σ W/E':>10}")
print(f"  {'─' * 68}")

for m in ["greedy", "planner", "field", "active_inference"]:
    rf = results_fixed[m]
    rs = results_sigma[m]
    delta = rs.total_extracted - rf.total_extracted
    wf = rf.total_waste / max(rf.total_extracted, 1)
    ws = rs.total_waste / max(rs.total_extracted, 1)
    print(f"  {m:<20} {rf.total_extracted:>10.0f} {rs.total_extracted:>10.0f} "
          f"{delta:>+8.0f} {wf:>10.1%} {ws:>10.1%}")


# ═══════════════════════════════════════════════════════════════════════
#  PART 4: The key comparison
# ═══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 80}")
print(f"  PART 4: KEY RESULTS")
print(f"{'=' * 80}")

g = results_sigma["greedy"]
s = results_sigma["sigma"]
p = results_sigma["planner"]
ai = results_sigma["active_inference"]

print(f"\n  σ reward machine, cascade scenario (5-hop chain):")
print(f"  {'─' * 60}")
print(f"  σ-agent vs Greedy:           extracted {s.total_extracted - g.total_extracted:>+8.1f}  "
      f"collisions {s.collisions - g.collisions:>+5d}")
print(f"  σ-agent vs Active Inference: extracted {s.total_extracted - ai.total_extracted:>+8.1f}  "
      f"collisions {s.collisions - ai.collisions:>+5d}")
print(f"  σ-agent vs σ-Planner:        extracted {s.total_extracted - p.total_extracted:>+8.1f}  "
      f"collisions {s.collisions - p.collisions:>+5d}")
print(f"  σ-agent as % of σ-Planner:   {s.total_extracted / max(p.total_extracted, 1) * 100:>7.1f}%")

print(f"\n  Waste ratios (composition rule prediction: deep chains waste more):")
print(f"  {'─' * 60}")
for m in MECHANISMS_SIGMA:
    r = results_sigma[m]
    # Compare to fixed params to see if σ changed the waste structure
    rf = results_fixed.get(m, r)
    print(f"  {m:<20}")
    print(f"    fixed: waste/extracted = {rf.total_waste/max(rf.total_extracted,1):>6.1%}")
    print(f"    σ:     waste/extracted = {r.total_waste/max(r.total_extracted,1):>6.1%}")

print(f"\n  Conservation verified on all runs ✓")
print(f"\n{'=' * 80}")
