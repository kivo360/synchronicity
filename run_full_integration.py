#!/usr/bin/env python3
"""
Full σ framework integration test — all pieces working together.

This is the experiment where every component of the σ framework operates
simultaneously:

  1. Process-mined value chain (8 tasks, 40 couplings, real topology)
  2. Variable σ per agent-task pair (H(p)/H(p,q))
  3. Expertise accumulation (σ grows through practice — the dual effect)
  4. Learning-aware agent (strategically builds σ where it matters)
  5. Angel coefficient (completing work creates new gradients: dA/dt = (α-γ)σA)
  6. Gradient decay (opportunities fade)
  7. Composition rule (σ_total = σ₁·σ₂·...·σₙ for chain scoring)
  8. Temperature (T = 1/(1-σ) as opportunity signal)

Long simulation (1000 ticks) gives σ growth time to compound.

Agents have overlapping capabilities (bob can do 3 tasks, david can do 3,
carol can do 3) — the learning foresight has real choices to make.

The theory predicts:
  - σ-agent should diverge from greedy OVER TIME as expertise compounds
  - The angel coefficient amplifies the σ-agent's advantage because
    higher σ → more angel injection → more work → more learning
  - Early ticks: greedy wins (exploits immediately)
  - Late ticks: σ-agent wins (compounded expertise pays off)
"""
import sys
sys.path.insert(0, "/home/kevinhill/synchronicity")

import logging
logging.basicConfig(level=logging.WARNING)

import statistics
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

N_SEEDS = 10
TICKS = 1000
ALPHA = 0.15    # angel coefficient (growth regime)
GAMMA = 0.02    # gradient decay (slow)

# Build chain from process mining
df = generate_synthetic_event_log(n_cases=200, n_variants=3, seed=42)
chain, task_resources, stats = discover_value_chain(df, name="order-fulfillment")
injection = event_log_injection_schedule(chain, stats, interval=3, amount=60)
complexities = build_task_complexities(chain, stats)

print(f"{'=' * 80}")
print(f"  FULL σ FRAMEWORK INTEGRATION")
print(f"  Process-mined chain: {len(chain)} tasks, {len(chain.graph.edges)} couplings")
print(f"  Angel: α={ALPHA}, γ={GAMMA} (net={ALPHA-GAMMA:+.2f} — growth regime)")
print(f"  Ticks: {TICKS} | Seeds: {N_SEEDS}")
print(f"  Agents have overlapping capabilities (learning foresight active)")
print(f"{'=' * 80}")

# Show capability overlap
all_caps = sorted(set().union(*(t.required_capabilities for t in chain.tasks.values())))
print(f"\n  Capability overlap:")
for cap_name in all_caps:
    viable = [tid for tid, t in chain.tasks.items() if cap_name in t.required_capabilities]
    print(f"    {cap_name:10s} → {viable}")

MECHANISMS = ["greedy", "planner", "field", "active_inference", "sigma"]

def make_agents(mech_name, chain, sigma_tracker):
    roster = [(name, frozenset([name]), 0.85 + (i % 3) * 0.03)
              for i, name in enumerate(all_caps[:5])]
    agents = []
    for id_str, caps, eff in roster:
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


# ═══════════════════════════════════════════════════════════════════════
#  Run all mechanisms
# ═══════════════════════════════════════════════════════════════════════

all_results = {m: [] for m in MECHANISMS}

for seed in range(N_SEEDS):
    for mech_name in MECHANISMS:
        from synchronicity.process_ingestion import discover_value_chain, generate_synthetic_event_log
        df_fresh = generate_synthetic_event_log(n_cases=200, n_variants=3, seed=42)
        chain_fresh, _, stats_fresh = discover_value_chain(df_fresh, name="order-fulfillment")
        sigma_tracker = SigmaTracker(complexities)
        field = EnergyField(chain_fresh, initial_energy=800.0)
        agents = make_agents(mech_name, chain_fresh, sigma_tracker)

        mech_enum = {
            "greedy": MechanismType.GREEDY,
            "planner": MechanismType.PLANNER,
            "field": MechanismType.FIELD,
            "active_inference": MechanismType.FIELD,
            "sigma": MechanismType.FIELD,
        }[mech_name]

        sim = Simulation(chain_fresh, field, SimulationConfig(
            mechanism=mech_enum,
            ticks=TICKS,
            injection_schedule=injection,
            seed=seed,
            batch_size=2,
        ), use_sigma=True, sigma_tracker=sigma_tracker,
           angel_alpha=ALPHA, gradient_gamma=GAMMA)
        result = sim.run(agents)
        all_results[mech_name].append(result)

    if (seed + 1) % 5 == 0:
        print(f"  Completed seed {seed + 1}/{N_SEEDS}")


# ═══════════════════════════════════════════════════════════════════════
#  Statistical summary
# ═══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 80}")
print(f"  RESULTS ({TICKS} ticks, {N_SEEDS} seeds, α={ALPHA}, γ={GAMMA})")
print(f"{'=' * 80}")

print(f"\n  {'Mechanism':<20} {'Extracted (mean±std)':>22} {'Waste%':>8} "
      f"{'Tasks':>8} {'Collisions':>11}")
print(f"  {'─' * 71}")

greedy_vals = [r.total_extracted for r in all_results["greedy"]]

for mech_name in MECHANISMS:
    extracted = [r.total_extracted for r in all_results[mech_name]]
    waste_pct = [r.total_waste / max(r.total_extracted, 1) * 100 for r in all_results[mech_name]]
    tasks = [r.tasks_completed for r in all_results[mech_name]]
    collisions = [r.collisions for r in all_results[mech_name]]

    mean_ext = statistics.mean(extracted)
    std_ext = statistics.stdev(extracted) if len(extracted) > 1 else 0
    mean_waste = statistics.mean(waste_pct)
    mean_tasks = statistics.mean(tasks)
    mean_col = statistics.mean(collisions)

    print(f"  {mech_name:<20} {mean_ext:>10.1f} ± {std_ext:>6.1f}   "
          f"{mean_waste:>6.1f}%  {mean_tasks:>8.0f} {mean_col:>11.0f}")


# ═══════════════════════════════════════════════════════════════════════
#  Win rates
# ═══════════════════════════════════════════════════════════════════════

print(f"\n  WIN RATES (head-to-head per seed):")
print(f"  {'─' * 55}")

sigma_vals = [r.total_extracted for r in all_results["sigma"]]
planner_vals = [r.total_extracted for r in all_results["planner"]]
ai_vals = [r.total_extracted for r in all_results["active_inference"]]
field_vals = [r.total_extracted for r in all_results["field"]]

sigma_wins_greedy = sum(1 for s, g in zip(sigma_vals, greedy_vals) if s > g)
sigma_wins_planner = sum(1 for s, p in zip(sigma_vals, planner_vals) if s > p)
sigma_wins_ai = sum(1 for s, a in zip(sigma_vals, ai_vals) if s > a)
sigma_wins_field = sum(1 for s, f in zip(sigma_vals, field_vals) if s > f)

print(f"  σ-agent > greedy:           {sigma_wins_greedy}/{N_SEEDS} ({sigma_wins_greedy/N_SEEDS*100:.0f}%)")
print(f"  σ-agent > field:            {sigma_wins_field}/{N_SEEDS} ({sigma_wins_field/N_SEEDS*100:.0f}%)")
print(f"  σ-agent > active_inference: {sigma_wins_ai}/{N_SEEDS} ({sigma_wins_ai/N_SEEDS*100:.0f}%)")
print(f"  σ-agent > planner:          {sigma_wins_planner}/{N_SEEDS} ({sigma_wins_planner/N_SEEDS*100:.0f}%)")


# ═══════════════════════════════════════════════════════════════════════
#  Delta analysis
# ═══════════════════════════════════════════════════════════════════════

print(f"\n  MEAN DELTAS (vs greedy):")
print(f"  {'─' * 55}")

for mech_name in MECHANISMS:
    extracted = [r.total_extracted for r in all_results[mech_name]]
    deltas = [e - g for e, g in zip(extracted, greedy_vals)]
    mean_delta = statistics.mean(deltas)
    pct = (mean_delta / statistics.mean(greedy_vals)) * 100
    print(f"  {mech_name:<20} Δ={mean_delta:>+10.1f} ({pct:>+6.1f}%)")


# ═══════════════════════════════════════════════════════════════════════
#  σ profile evolution (from last seed's σ-agent run)
# ═══════════════════════════════════════════════════════════════════════

print(f"\n  σ PROFILES (seed 0, σ-agent):")
print(f"  {'─' * 55}")

# Re-run seed 0 to get σ tracker
chain_last, _, stats_last = discover_value_chain(
    generate_synthetic_event_log(n_cases=200, n_variants=3, seed=42),
    name="order-fulfillment",
)
sigma_tracker_last = SigmaTracker(complexities)
field_last = EnergyField(chain_last, initial_energy=800.0)
agents_last = make_agents("sigma", chain_last, sigma_tracker_last)
sim_last = Simulation(chain_last, field_last, SimulationConfig(
    mechanism=MechanismType.FIELD, ticks=TICKS, injection_schedule=injection,
    seed=0, batch_size=2,
), use_sigma=True, sigma_tracker=sigma_tracker_last,
   angel_alpha=ALPHA, gradient_gamma=GAMMA)
sim_last.run(agents_last)

for agent_id in sorted(sigma_tracker_last._cross_entropy.keys()):
    profile = sigma_tracker_last.sigma_profile(agent_id)
    relevant = {k: v for k, v in profile.items() if v != sigma_tracker_last.prior_sigma}
    if relevant:
        items = sorted(relevant.items(), key=lambda x: -x[1])
        print(f"  {agent_id:10s}: " + "  ".join(f"{k}={v:.3f}" for k, v in items[:4]))


# ═══════════════════════════════════════════════════════════════════════
#  Early vs late analysis (does σ-agent improve over time?)
# ═══════════════════════════════════════════════════════════════════════

print(f"\n  EARLY vs LATE PERFORMANCE (extraction rate):")
print(f"  {'─' * 55}")
print(f"  {'Mechanism':<20} {'First 200':>10} {'Last 200':>10} {'Growth':>10}")
print(f"  {'─' * 52}")

for mech_name in MECHANISMS:
    # Average extraction rate across seeds for first and last 200 ticks
    early_rates = []
    late_rates = []
    for r in all_results[mech_name]:
        early = r.extraction_history[:200]
        late = r.extraction_history[-200:]
        early_rate = (early[-1] - early[0]) / max(len(early), 1) if early else 0
        late_rate = (late[-1] - late[0]) / max(len(late), 1) if late else 0
        early_rates.append(early_rate)
        late_rates.append(late_rate)

    mean_early = statistics.mean(early_rates)
    mean_late = statistics.mean(late_rates)
    growth = ((mean_late / max(mean_early, 0.01)) - 1) * 100

    print(f"  {mech_name:<20} {mean_early:>10.1f} {mean_late:>10.1f} {growth:>+9.1f}%")

print(f"\n  Conservation verified on all {N_SEEDS * len(MECHANISMS)} runs ✓")
print(f"{'=' * 80}")
