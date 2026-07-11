# Synchronicity

Energy-based incentive fields for multi-agent economic coordination.

Synchronicity models economic coordination as a physical system. Tasks are
nodes in an energy field. Value chains are edges. Agents read the field,
act, and the field updates according to thermodynamic identities —
conservation of energy, entropy increase, capacity saturation.

## Core Claim

Agents navigating an energy-based incentive field self-organize into more
efficient configurations than pure greedy optimization, approaching
central-planner efficiency while maintaining local autonomy.

## Quick Start

```bash
pip install -e .

# Run the 5-mechanism comparison on a process-mined value chain
python run_full_comparison.py

# Run the batch-size sweep across 4 scenarios
python run_full_analysis.py

# Generate visualization data and serve the viewer
python run_recording.py
python -m http.server 8765
# Open http://localhost:8765/viz.html
```

## Architecture

```
┌───────────────────────────────────────────────┐
│            COORDINATION LAYER                  │
│   5 agent mechanisms (agents.py,               │
│   active_inference.py, llm_agent.py)           │
├───────────────────────────────────────────────┤
│            DISCOVERY LAYER                     │
│   Auto-discover value chains from              │
│   business event logs (process_ingestion.py)   │
│   using PM4Py process mining                   │
├───────────────────────────────────────────────┤
│            ENERGY FIELD ENGINE                 │
│   Conservation laws, entropy, couplings,       │
│   value chain topology (field.py,              │
│   reward_machine.py, value_chain.py)           │
└───────────────────────────────────────────────┘
```

## Physics Invariants

Enforced as runtime assertions on every field update:

1. **Conservation of energy** — `injected = field_energy + extracted + waste`
2. **Entropy increase** — waste only accumulates, never decreases
3. **Capacity saturation** — overflow routed elsewhere, never destroyed

## Agent Mechanisms

| Mechanism | Scoring | Collisions | Description |
|-----------|---------|------------|-------------|
| Greedy | Raw energy | High | Baseline: self-interested, no coordination |
| Planner | Optimal assignment | 0 | Upper bound: central control with full knowledge |
| Field | Heuristic + learned competition | Medium | Synchronicity: energy field navigation |
| Active Inference | FEP expected free energy | **Low** | Friston's free energy principle |
| LLM | Natural language reasoning | — | Real Hermes agent reading the field |

Key result: Active inference reduces collisions by **73% vs greedy** while
maintaining 99.7% of planner throughput.

## Modules

| Module | Purpose |
|--------|---------|
| `value_chain.py` | Static topology: tasks, couplings, DAG |
| `field.py` | Dynamic state: energy allocation, conservation |
| `reward_machine.py` | Update rules: entropy, extraction, propagation |
| `agents.py` | Greedy, planner, field agents + softmax sampling |
| `active_inference.py` | FEP-based agent: pragmatic + epistemic - ambiguity |
| `llm_agent.py` | Hermes/OpenAI LLM adapter |
| `process_ingestion.py` | PM4Py event log → auto-discovered value chains |
| `simulation.py` | Batch execution, metrics, comparison harness |
| `scenarios.py` | 4 hand-coded scenarios + registry |
| `recorder.py` | Per-tick recording for visualization |

## Runners

| Script | Purpose |
|--------|---------|
| `run_full_comparison.py` | 5-mechanism comparison on process-mined chain |
| `run_full_analysis.py` | Multi-scenario + parameter sweep |
| `run_batch_sweep.py` | Batch-size sweep on trash-cleanup scenario |
| `run_recording.py` | Record runs for visualization |

## Process-Mining Integration

Auto-discover value chains from real business event logs:

```python
from synchronicity.process_ingestion import (
    generate_synthetic_event_log, discover_value_chain,
)

# Generate or load a PM4Py-format event log
df = generate_synthetic_event_log(n_cases=200, n_variants=3)

# Discover the value chain topology
chain, task_resources, stats = discover_value_chain(df)

# 8 tasks, 40 couplings auto-discovered from 1066 events
```

## Status

**Foundation complete.** Energy field engine, conservation invariants,
process-mining ingestion, 5 agent mechanisms (including real LLM),
simulation harness, parameter sweep, and interactive visualization.

**Active research question:** The efficiency ratio (~80-85%) is structurally
determined by the reward machine parameters, not agent behavior. The metrics
that differentiate coordination quality are collisions, tasks completed,
and total extracted value. The field equations (in development) will replace
the heuristic and active-inference scoring with physics-derived gradients.

## License

MIT
