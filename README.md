# Synchronicity

Energy-based incentive fields for multi-agent economic coordination.

Synchronicity models economic coordination as a physical system. Tasks are
nodes in an energy field. Value chains are edges. Agents read the field,
act, and the field updates according to thermodynamic identities —
conservation of energy, entropy increase, capacity saturation.

## Why

Current multi-agent frameworks (CrewAI, AutoGen, LangGraph) have no
principled model of incentives or rewards. Synchronicity provides one,
grounded in physics rather than heuristics.

The core claim: agents navigating an energy-based incentive field
self-organize into more efficient configurations than pure greedy
optimization, approaching central-planner efficiency while maintaining
local autonomy and adaptability.

## Architecture

```
┌──────────────────────────────────────┐
│         COORDINATION LAYER           │   Multiple agents reading the
│   (simulation.py, agents.py)         │   same field, acting independently
├──────────────────────────────────────┤
│         AGENT INTERFACE              │   Any agent reads field → reasons
│   (field.snapshot → FieldSnapshot)   │   → acts → reports outcome
├──────────────────────────────────────┤
│         ENERGY FIELD ENGINE          │   Pure math. Conservation laws,
│   (field.py, reward_machine.py,      │   entropy, value chain topology.
│    value_chain.py)                   │   Agent-agnostic. The physics.
└──────────────────────────────────────┘
```

## Physics Invariants

The simulation enforces three laws as runtime assertions:

1. **Conservation of energy** — `injected = field_energy + extracted + waste`.
   Verified on every field update. If this fails, the formalization is wrong.

2. **Entropy increase** — waste energy only accumulates, never decreases.
   Every transaction has an irrecoverable cost.

3. **Capacity saturation** — no task node can hold more energy than its
   capacity. Overflow is routed elsewhere, never destroyed.

## Quick Start

```bash
pip install -e .
python run_comparison.py
```

## Modules

| Module | Purpose |
|--------|---------|
| `value_chain.py` | Static topology: tasks, couplings, value chain DAG |
| `field.py` | Dynamic state: per-task energy, conservation enforcement |
| `reward_machine.py` | Update rules: entropy, extraction, propagation |
| `agents.py` | Agent protocol + three baselines (greedy, planner, field) |
| `simulation.py` | Experiment harness: run and compare mechanisms |
| `scenarios.py` | Demo scenarios (trash-cleanup value chain) |

## Status

**Foundation complete.** The energy field engine, conservation invariants,
agent protocol, and simulation harness are working. Conservation is
verified across 200-tick runs with thousands of events.

**Active research question:** the current simulation does not yet
differentiate the three mechanisms — all converge to ~85% efficiency.
This is because the sequential execution model gives greedy agents
implicit coordination (they see updated field state between actions).
The next step is to introduce simultaneous commitment (agents commit
before seeing others' actions) and bottleneck scenarios where
downstream awareness genuinely matters.

## License

MIT
