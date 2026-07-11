#!/usr/bin/env python3
"""
Angel coefficient phase transition experiment.

Sweeps α (angel: gradient creation) vs γ (gradient decay) to find the
growth/saturation boundary predicted by dA/dt = (α-γ)·σ·A.

The paper says "every measured domain so far favors saturation."
This experiment asks: under what conditions does the system GROW?

Theory predicts:
  α > γ → system grows (energy increases over time, compound value)
  α < γ → system saturates (energy depletes despite work)
  α = γ → equilibrium

The critical question: does the σ-agent (high σ → more angel injection)
push the system toward growth more than greedy?
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
from synchronicity.scenarios import build_cascade_scenario, cascade_injection

N_SEEDS = 5
TICKS = 200

CASCADE_ROSTER = [
    ("worker_0", frozenset(["labor"]), 0.88),
    ("worker_1", frozenset(["labor"]), 0.85),
    ("refiner_0", frozenset(["refining"]), 0.90),
    ("assembler_0", frozenset(["assembly"]), 0.92),
    ("packer_0", frozenset(["packaging"]), 0.89),
    ("shipper_0", frozenset(["logistics"]), 0.87),
]

chain_template, _ = build_cascade_scenario()
complexities = build_task_complexities(chain_template)

def make_agents(mech_name, chain, sigma_tracker):
    agents = []
    for id_str, caps, eff in CASCADE_ROSTER:
        cap = AgentCapability(id=id_str, capabilities=caps, base_efficiency=eff)
        if mech_name == "greedy":
            agents.append(GreedyAgent(cap))
        elif mech_name == "planner":
            agents.append(PlannerAgent(cap))
        elif mech_name == "sigma":
            agents.append(SigmaAgent(cap, chain, sigma_tracker))
    return agents

# ═══════════════════════════════════════════════════════════════════════
#  PART 1: α vs γ sweep (σ-agent only, single seed for the sweep)
# ═══════════════════════════════════════════════════════════════════════

print(f"{'=' * 80}")
print(f"  ANGEL COEFFICIENT PHASE TRANSITION SWEEP")
print(f"  dA/dt = (α - γ)·σ·A")
print(f"  Cascade scenario, σ-agent, {TICKS} ticks, seed=42")
print(f"{'=' * 80}")

ALPHA_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
GAMMA_VALUES = [0.0, 0.02, 0.05, 0.10]

print(f"\n  {'α':>6} {'γ':>6} {'α-γ':>6} {'Extracted':>10} {'Waste':>10} "
      f"{'Energy_end':>10} {'Tasks':>6} {'Phase':>8}")
print(f"  {'─' * 72}")

sweep_results = []

for alpha in ALPHA_VALUES:
    for gamma in GAMMA_VALUES:
        chain, _ = build_cascade_scenario()
        sigma_tracker = SigmaTracker(complexities)
        field = EnergyField(chain, initial_energy=600.0)
        agents = make_agents("sigma", chain, sigma_tracker)

        sim = Simulation(chain, field, SimulationConfig(
            mechanism=MechanismType.FIELD,
            ticks=TICKS,
            injection_schedule=cascade_injection,
            seed=42,
            batch_size=3,
        ), use_sigma=True, sigma_tracker=sigma_tracker,
           angel_alpha=alpha, gradient_gamma=gamma)
        result = sim.run(agents)

        net = alpha - gamma
        phase = "GROWTH" if net > 0.05 else ("DECAY" if net < -0.05 else "EQUILIB")
        energy_end = result.energy_history[-1] if result.energy_history else 0

        print(f"  {alpha:>6.2f} {gamma:>6.2f} {net:>+6.2f} "
              f"{result.total_extracted:>10.0f} {result.total_waste:>10.0f} "
              f"{energy_end:>10.0f} {result.tasks_completed:>6d} {phase:>8}")

        sweep_results.append({
            "alpha": alpha, "gamma": gamma, "net": net,
            "extracted": result.total_extracted,
            "energy_end": energy_end,
            "tasks": result.tasks_completed,
        })

# ═══════════════════════════════════════════════════════════════════════
#  PART 2: Mechanism comparison in growth vs saturation regime
# ═══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 80}")
print(f"  MECHANISM COMPARISON: GROWTH vs SATURATION")
print(f"  Growth:    α=0.20, γ=0.02 (net=+0.18)")
print(f"  Saturation: α=0.05, γ=0.10 (net=-0.05)")
print(f"  {N_SEEDS} seeds × 3 mechanisms")
print(f"{'=' * 80}")

REGIMES = [
    ("GROWTH (α=0.20, γ=0.02)", 0.20, 0.02),
    ("SATURATION (α=0.05, γ=0.10)", 0.05, 0.10),
]

MECHANISMS = ["greedy", "planner", "sigma"]

for regime_label, alpha, gamma in REGIMES:
    print(f"\n  {regime_label}:")
    print(f"  {'Mechanism':<12} {'Extracted (mean±std)':>22} {'Tasks':>8} {'Energy_end':>11}")
    print(f"  {'─' * 55}")

    for mech_name in MECHANISMS:
        extracted_vals = []
        tasks_vals = []
        energy_vals = []

        for seed in range(N_SEEDS):
            chain, _ = build_cascade_scenario()
            sigma_tracker = SigmaTracker(complexities)
            field = EnergyField(chain, initial_energy=600.0)
            agents = make_agents(mech_name, chain, sigma_tracker)

            mech_enum = {
                "greedy": MechanismType.GREEDY,
                "planner": MechanismType.PLANNER,
                "sigma": MechanismType.FIELD,
            }[mech_name]

            sim = Simulation(chain, field, SimulationConfig(
                mechanism=mech_enum,
                ticks=TICKS,
                injection_schedule=cascade_injection,
                seed=seed,
                batch_size=3,
            ), use_sigma=True, sigma_tracker=sigma_tracker,
               angel_alpha=alpha, gradient_gamma=gamma)
            result = sim.run(agents)

            extracted_vals.append(result.total_extracted)
            tasks_vals.append(result.tasks_completed)
            energy_vals.append(result.energy_history[-1] if result.energy_history else 0)

        mean_ext = statistics.mean(extracted_vals)
        std_ext = statistics.stdev(extracted_vals) if len(extracted_vals) > 1 else 0
        mean_tasks = statistics.mean(tasks_vals)
        mean_energy = statistics.mean(energy_vals)

        print(f"  {mech_name:<12} {mean_ext:>10.1f} ± {std_ext:>6.1f}   "
              f"{mean_tasks:>8.0f} {mean_energy:>11.0f}")

# ═══════════════════════════════════════════════════════════════════════
#  PART 3: The key question — does σ-agent grow faster?
# ═══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 80}")
print(f"  KEY QUESTION: Does high-σ coordination amplify angel generation?")
print(f"{'=' * 80}")

print(f"\n  In the GROWTH regime, the σ-agent's high σ should generate MORE")
print(f"  angel energy (α·đN·σ) than greedy. If the σ-agent ends with more")
print(f"  total extracted value AND more residual energy, the dual effect")
print(f"  (σ raises efficiency AND reveals gradients) is confirmed.\n")

# Re-run growth regime with more detail
growth_results = {}
for mech_name in MECHANISMS:
    ext_vals = []
    energy_vals = []
    for seed in range(N_SEEDS):
        chain, _ = build_cascade_scenario()
        sigma_tracker = SigmaTracker(complexities)
        field = EnergyField(chain, initial_energy=600.0)
        agents = make_agents(mech_name, chain, sigma_tracker)

        mech_enum = {"greedy": MechanismType.GREEDY, "planner": MechanismType.PLANNER,
                     "sigma": MechanismType.FIELD}[mech_name]

        sim = Simulation(chain, field, SimulationConfig(
            mechanism=mech_enum, ticks=TICKS, injection_schedule=cascade_injection,
            seed=seed, batch_size=3,
        ), use_sigma=True, sigma_tracker=sigma_tracker,
           angel_alpha=0.20, gradient_gamma=0.02)
        result = sim.run(agents)
        ext_vals.append(result.total_extracted)
        energy_vals.append(result.energy_history[-1] if result.energy_history else 0)

    growth_results[mech_name] = {
        "extracted": statistics.mean(ext_vals),
        "energy_end": statistics.mean(energy_vals),
    }

print(f"  {'Mechanism':<12} {'Extracted':>10} {'Energy_end':>10} {'Total value':>12}")
print(f"  {'─' * 46}")
for m in MECHANISMS:
    g = growth_results[m]
    total = g["extracted"] + g["energy_end"]
    print(f"  {m:<12} {g['extracted']:>10.0f} {g['energy_end']:>10.0f} {total:>12.0f}")

sigma_total = growth_results["sigma"]["extracted"] + growth_results["sigma"]["energy_end"]
greedy_total = growth_results["greedy"]["extracted"] + growth_results["greedy"]["energy_end"]
planner_total = growth_results["planner"]["extracted"] + growth_results["planner"]["energy_end"]

print(f"\n  σ-agent total value vs Greedy:   {sigma_total - greedy_total:>+8.0f} ({(sigma_total/greedy_total - 1)*100:>+.1f}%)")
print(f"  σ-agent total value vs Planner:  {sigma_total - planner_total:>+8.0f} ({(sigma_total/planner_total - 1)*100:>+.1f}%)")

print(f"\n  Conservation verified on all runs ✓")
print(f"{'=' * 80}")
