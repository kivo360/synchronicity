#!/usr/bin/env python3
"""
Record simulation runs for visualization.

Runs greedy, planner, and field mechanisms on the trash-cleanup scenario,
records detailed per-tick state, and exports to viz_data.json.

The HTML viewer (viz.html) loads this JSON and animates the three runs
side by side.
"""
import sys
import json
sys.path.insert(0, "/home/kevinhill/synchronicity")

import logging
logging.basicConfig(level=logging.WARNING)

from synchronicity.simulation import Simulation, SimulationConfig, MechanismType
from synchronicity.reward_machine import FieldEvent, EventType
from synchronicity.recorder import SimulationRecorder
from synchronicity.scenarios import build_trash_cleanup_scenario, make_agents_for_mechanism

TICKS = 200
SEED = 42
BATCH_SIZE = 3  # medium collision pressure — most interesting visually

def injection_schedule(tick: int) -> list:
    if tick % 3 == 0:
        return [
            FieldEvent(event_type=EventType.CAPITAL_INJECTED, task_id="collect_north", amount=80),
            FieldEvent(event_type=EventType.CAPITAL_INJECTED, task_id="collect_south", amount=80),
            FieldEvent(event_type=EventType.CAPITAL_INJECTED, task_id="park_cleanup", amount=50),
        ]
    return []

# Task layout positions for visualization (left-to-right value chain flow)
TASK_POSITIONS = {
    "collect_north":  {"x": 0.08, "y": 0.20},
    "collect_south":  {"x": 0.08, "y": 0.50},
    "park_cleanup":   {"x": 0.08, "y": 0.80},
    "sort_center":    {"x": 0.32, "y": 0.50},
    "compost":        {"x": 0.56, "y": 0.25},
    "recycling":      {"x": 0.56, "y": 0.75},
    "market_stall":   {"x": 0.82, "y": 0.25},
    "wholesale":      {"x": 0.82, "y": 0.75},
}

def get_chain_metadata(chain):
    """Extract positions, types, and coupling info for viz."""
    task_types = {}
    for tid in chain.all_task_ids():
        task_types[tid] = chain.get_task(tid).task_type.value

    couplings = []
    for source in chain.all_task_ids():
        for target, coeff in chain.downstream(source):
            couplings.append({
                "source": source,
                "target": target,
                "coefficient": coeff,
            })

    return task_types, couplings

# Run all three mechanisms
runs = {}

for mechanism in [MechanismType.GREEDY, MechanismType.PLANNER, MechanismType.FIELD]:
    chain, field = build_trash_cleanup_scenario()
    agents = make_agents_for_mechanism(mechanism, chain)

    task_types, couplings = get_chain_metadata(chain)
    recorder = SimulationRecorder(TASK_POSITIONS, task_types, couplings)

    sim = Simulation(chain, field, SimulationConfig(
        mechanism=mechanism,
        ticks=TICKS,
        injection_schedule=injection_schedule,
        seed=SEED,
        batch_size=BATCH_SIZE,
    ), recorder=recorder)

    result = sim.run(agents)
    runs[mechanism.value] = recorder.to_dict(mechanism.value, result)

    print(f"  {mechanism.value:8s}: extracted={result.total_extracted:.0f}  "
          f"collisions={result.collisions}  tasks={result.tasks_completed}")

# Export
output_path = "/home/kevinhill/synchronicity/viz_data.json"
with open(output_path, "w") as f:
    json.dump(runs, f)

print(f"\nExported {len(runs)} runs to {output_path}")
print(f"File size: {len(json.dumps(runs))} bytes")
