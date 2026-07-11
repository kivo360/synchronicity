"""
Scenarios for the Synchronicity simulation.

Each scenario defines a value chain topology, agent roster, and energy
parameters. The scenario design is critical to the experiment: the value
chain must contain structures where coordination strategy actually matters.

Key design principle: simple linear chains (A → B → C) make greedy look
good because there's always an obvious next task. The Synchronicity
mechanism's advantage shows when:

  1. Branching value chains exist (doing task A could feed B OR C)
  2. Bridge tasks exist (doing A unlocks D, E, F downstream — but only
     if you know to look 2+ hops ahead)
  3. Capacity bottlenecks exist (flooding one task wastes overflow)
  4. Capability constraints exist (not every agent can do every task)
"""

from __future__ import annotations

from synchronicity.field import EnergyField
from synchronicity.reward_machine import FieldEvent, EventType
from synchronicity.value_chain import ValueChain, Task, Coupling, TaskType
from synchronicity.agents import (
    AgentCapability,
    GreedyAgent,
    FieldAgent,
    PlannerAgent,
)


def build_trash_cleanup_scenario() -> tuple[ValueChain, EnergyField]:
    """Trash cleanup value chain — the canonical Synchronicity example.

    Topology:
        collect_north ─┐
        collect_south ─┼→ sort_center ─┬→ compost ─→ market_stall
        park_cleanup  ─┘               └→ recycling → wholesale

    This is a branching chain: sort_center is a bridge node whose two
    downstream paths (composting vs recycling) serve different markets.
    Agents that only look at immediate energy (greedy) will flood whichever
    path currently has the most energy, causing waste through capacity
    overflow. Agents that read the field structurally will balance the two
    paths and avoid bottlenecks.

    The sorting task is the strategic decision point: completing it routes
    energy to BOTH downstream paths. A greedy agent sees sorting as just
    another task with some energy. A field-aware agent sees it as a
    multiplier — one completion feeds two value chains.
    """
    chain = ValueChain("trash-cleanup")

    tasks = [
        Task(
            id="collect_north",
            task_type=TaskType.COLLECTION,
            location="north",
            energy_capacity=120.0,
            required_capabilities=frozenset(["labor"]),
        ),
        Task(
            id="collect_south",
            task_type=TaskType.COLLECTION,
            location="south",
            energy_capacity=120.0,
            required_capabilities=frozenset(["labor"]),
        ),
        Task(
            id="park_cleanup",
            task_type=TaskType.SERVICE,
            location="central_park",
            energy_capacity=80.0,
            required_capabilities=frozenset(["labor"]),
        ),
        Task(
            id="sort_center",
            task_type=TaskType.PROCESSING,
            location="center",
            energy_capacity=150.0,
            required_capabilities=frozenset(["sorting"]),
        ),
        Task(
            id="compost",
            task_type=TaskType.PROCESSING,
            location="center",
            energy_capacity=100.0,
            required_capabilities=frozenset(["processing"]),
        ),
        Task(
            id="recycling",
            task_type=TaskType.PROCESSING,
            location="east",
            energy_capacity=100.0,
            required_capabilities=frozenset(["processing"]),
        ),
        Task(
            id="market_stall",
            task_type=TaskType.RETAIL,
            location="downtown",
            energy_capacity=80.0,
            required_capabilities=frozenset(["sales"]),
        ),
        Task(
            id="wholesale",
            task_type=TaskType.DISTRIBUTION,
            location="port",
            energy_capacity=80.0,
            required_capabilities=frozenset(["sales"]),
        ),
    ]

    for task in tasks:
        chain.add_task(task)

    couplings = [
        # Collection feeds sorting
        Coupling("collect_north", "sort_center", coefficient=0.8),
        Coupling("collect_south", "sort_center", coefficient=0.8),
        Coupling("park_cleanup", "sort_center", coefficient=0.6),

        # Sorting branches into two paths
        Coupling("sort_center", "compost", coefficient=0.5),
        Coupling("sort_center", "recycling", coefficient=0.5),

        # Each path feeds its own market
        Coupling("compost", "market_stall", coefficient=1.0),
        Coupling("recycling", "wholesale", coefficient=1.0),
    ]

    for coupling in couplings:
        chain.add_coupling(coupling)

    field = EnergyField(chain, initial_energy=1200.0)
    return chain, field


def make_agents_for_mechanism(mechanism, chain: ValueChain) -> list:
    """Create agents pre-configured for a specific mechanism and chain.

    Agent roster (6 agents, 3 capability types):
      - 3 laborers (can collect + park cleanup)
      - 2 specialists (can sort + process)
      - 1 vendor (can run market stall + wholesale)

    The capability distribution is intentionally constrained: only 2 agents
    can sort, which is the bridge task. This creates a strategic decision —
    if both specialists are busy processing downstream, sorting (and thus
    the whole value chain) stalls.
    """
    agents = []

    agent_class = {
        "greedy": GreedyAgent,
        "field": FieldAgent,
        "planner": PlannerAgent,
    }[mechanism.value if hasattr(mechanism, "value") else mechanism]

    for i in range(3):
        cap = AgentCapability(
            id=f"laborer_{i}",
            capabilities=frozenset(["labor"]),
            base_efficiency=0.85 + (i * 0.03),
        )
        if agent_class is FieldAgent:
            agents.append(FieldAgent(cap, chain))
        else:
            agents.append(agent_class(cap))

    for i in range(2):
        cap = AgentCapability(
            id=f"specialist_{i}",
            capabilities=frozenset(["sorting", "processing"]),
            base_efficiency=0.90 + (i * 0.02),
        )
        if agent_class is FieldAgent:
            agents.append(FieldAgent(cap, chain))
        else:
            agents.append(agent_class(cap))

    cap = AgentCapability(
        id="vendor_0",
        capabilities=frozenset(["sales"]),
        base_efficiency=0.92,
    )
    if agent_class is FieldAgent:
        agents.append(FieldAgent(cap, chain))
    else:
        agents.append(agent_class(cap))

    return agents
