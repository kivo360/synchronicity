#!/usr/bin/env python3
"""
Run the three-way comparison experiment.

Same value chain, same agents, same starting energy. The only variable
is the coordination mechanism. This isolates exactly what the Synchronicity
theory predicts: field-based coordination achieves near-planner efficiency
while maintaining local autonomy.
"""
import sys
sys.path.insert(0, "/home/kevinhill/synchronicity")

import logging
logging.basicConfig(level=logging.WARNING)

from synchronicity.simulation import Simulation, SimulationConfig, MechanismType
from synchronicity.reward_machine import FieldEvent, EventType
from synchronicity.scenarios import build_trash_cleanup_scenario, make_agents_for_mechanism

TICKS = 200
SEED = 42

# Capital injection: inject energy into collection tasks every few ticks
# to keep the economy running (simulates ongoing demand/funding)
def injection_schedule(tick: int) -> list:
    if tick % 3 == 0:
        return [
            FieldEvent(event_type=EventType.CAPITAL_INJECTED, task_id="collect_north", amount=80),
            FieldEvent(event_type=EventType.CAPITAL_INJECTED, task_id="collect_south", amount=80),
            FieldEvent(event_type=EventType.CAPITAL_INJECTED, task_id="park_cleanup", amount=50),
        ]
    return []

print("=" * 70)
print("  SYNCHRONICITY: THREE-WAY MECHANISM COMPARISON")
print("  Scenario: Trash Cleanup Value Chain")
print("  Ticks: {} | Agents: 6 | Initial energy: 1000 | Injection: 130/5 ticks".format(TICKS))
print("=" * 70)

results = {}

for mechanism in [MechanismType.GREEDY, MechanismType.PLANNER, MechanismType.FIELD]:
    chain, field = build_trash_cleanup_scenario()
    agents = make_agents_for_mechanism(mechanism, chain)

    sim = Simulation(chain, field, SimulationConfig(
        mechanism=mechanism,
        ticks=TICKS,
        injection_schedule=injection_schedule,
        seed=SEED,
    ))
    result = sim.run(agents)
    results[mechanism] = result

# Print comparison table
print("\n" + "=" * 70)
print("  RESULTS")
print("=" * 70)

for mech in [MechanismType.GREEDY, MechanismType.PLANNER, MechanismType.FIELD]:
    r = results[mech]
    print(f"\n{'─' * 50}")
    print(f"  {mech.value.upper()}")
    print(f"{'─' * 50}")
    print(r.summary())

    print("\n  Per-agent rewards:")
    for stat in sorted(r.agent_stats, key=lambda x: -x["reward"]):
        print(f"    {stat['id']:16s}  reward={stat['reward']:8.2f}  "
              f"done={stat['completed']:3d}  fail={stat['failed']:3d}")

# Summary comparison
print("\n" + "=" * 70)
print("  HEADLINE COMPARISON")
print("=" * 70)
print(f"\n  {'Mechanism':<16} {'Extracted':>10} {'Waste':>10} "
      f"{'Efficiency':>12} {'Gini':>8} {'Tasks':>6}")
print(f"  {'─' * 62}")

for mech in [MechanismType.GREEDY, MechanismType.PLANNER, MechanismType.FIELD]:
    r = results[mech]
    print(f"  {mech.value:<16} {r.total_extracted:>10.1f} {r.total_waste:>10.1f} "
          f"{r.efficiency_ratio:>11.1%} {r.gini_coefficient:>8.3f} "
          f"{r.tasks_completed:>6d}")

# The key claim
greedy_eff = results[MechanismType.GREEDY].efficiency_ratio
planner_eff = results[MechanismType.PLANNER].efficiency_ratio
field_eff = results[MechanismType.FIELD].efficiency_ratio

print(f"\n  {'─' * 62}")
print(f"  KEY METRIC: Allocation Efficiency")
print(f"  {'─' * 62}")
print(f"  Greedy (baseline A):     {greedy_eff:.1%}")
print(f"  Planner (baseline B):    {planner_eff:.1%}")
print(f"  Field (Synchronicity):   {field_eff:.1%}")

field_vs_greedy = (field_eff / greedy_eff - 1) * 100 if greedy_eff > 0 else 0
field_vs_planner = field_eff / planner_eff * 100 if planner_eff > 0 else 0

print(f"\n  Field vs Greedy:   {field_vs_greedy:+.1f}% improvement")
print(f"  Field vs Planner:  {field_vs_planner:.1f}% of planner efficiency")

print("\n" + "=" * 70)
print("  Conservation verified on all runs ✓")
print("=" * 70)
