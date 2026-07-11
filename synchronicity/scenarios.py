"""
Scenarios for the Synchronicity simulation.

Each scenario defines a value chain topology, agent roster, and energy
parameters. The scenario design is critical to the experiment: the value
chain must contain structures where coordination strategy actually matters.

Three scenarios, each stress-testing a different coordination property:

  1. Crossroads  — competing bottlenecks: shared specialists must balance
     two parallel value chains. Greedy floods one chain; field should split.

  2. Cascade     — deep multi-hop chain (5+ stages). Greedy clumps at the
     front; field agents should pre-position downstream before energy arrives.

  3. Burst       — temporal dynamics: energy arrives in waves at different
     nodes. Greedy floods peaks; field should anticipate and smooth.

Design principle: a constraint BINDS when multiple agents need the same
scarce resource, the capacity is tight relative to demand, and failing to
access it has cascading downstream effects. Every scenario below has at
least one binding constraint.
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


# ═══════════════════════════════════════════════════════════════════════
#  SCENARIO 1: CROSSROADS
# ═══════════════════════════════════════════════════════════════════════

def build_crossroads_scenario() -> tuple[ValueChain, EnergyField]:
    """Two parallel value chains sharing one bottleneck specialist.

    Topology:
        collect_A → sort_A → process_A → retail_A
        collect_B → sort_B → process_B → retail_B
                            ↑
                     BOTH sort tasks need the SAME specialist
                     (only 1 specialist agent in the roster)

    Binding constraint: 1 specialist must alternate between sort_A and
    sort_B. If it clumps on one chain (greedy), the other chain starves
    and its collected energy decays (entropy). If it balances (field),
    both chains flow.

    This is the canonical "shared resource scheduling" problem.
    """
    chain = ValueChain("crossroads")

    tasks = [
        # Chain A
        Task(id="collect_A", task_type=TaskType.COLLECTION,
             location="A", energy_capacity=80.0,
             required_capabilities=frozenset(["labor"])),
        Task(id="sort_A", task_type=TaskType.PROCESSING,
             location="A", energy_capacity=40.0,
             required_capabilities=frozenset(["sorting"])),
        Task(id="process_A", task_type=TaskType.PROCESSING,
             location="A", energy_capacity=60.0,
             required_capabilities=frozenset(["processing"])),
        Task(id="retail_A", task_type=TaskType.RETAIL,
             location="A", energy_capacity=40.0,
             required_capabilities=frozenset(["sales"])),

        # Chain B (parallel)
        Task(id="collect_B", task_type=TaskType.COLLECTION,
             location="B", energy_capacity=80.0,
             required_capabilities=frozenset(["labor"])),
        Task(id="sort_B", task_type=TaskType.PROCESSING,
             location="B", energy_capacity=40.0,
             required_capabilities=frozenset(["sorting"])),
        Task(id="process_B", task_type=TaskType.PROCESSING,
             location="B", energy_capacity=60.0,
             required_capabilities=frozenset(["processing"])),
        Task(id="retail_B", task_type=TaskType.RETAIL,
             location="B", energy_capacity=40.0,
             required_capabilities=frozenset(["sales"])),
    ]

    for t in tasks:
        chain.add_task(t)

    couplings = [
        # Chain A
        Coupling("collect_A", "sort_A", coefficient=0.9),
        Coupling("sort_A", "process_A", coefficient=0.9),
        Coupling("process_A", "retail_A", coefficient=1.0),
        # Chain B
        Coupling("collect_B", "sort_B", coefficient=0.9),
        Coupling("sort_B", "process_B", coefficient=0.9),
        Coupling("process_B", "retail_B", coefficient=1.0),
    ]
    for c in couplings:
        chain.add_coupling(c)

    field = EnergyField(chain, initial_energy=800.0)
    return chain, field


CROSSROADS_POSITIONS = {
    "collect_A": {"x": 0.08, "y": 0.20},
    "sort_A":    {"x": 0.32, "y": 0.20},
    "process_A": {"x": 0.56, "y": 0.20},
    "retail_A":  {"x": 0.82, "y": 0.20},
    "collect_B": {"x": 0.08, "y": 0.75},
    "sort_B":    {"x": 0.32, "y": 0.75},
    "process_B": {"x": 0.56, "y": 0.75},
    "retail_B":  {"x": 0.82, "y": 0.75},
}


def make_crossroads_agents(mechanism, chain: ValueChain) -> list:
    """Crossroads agent roster: 4 agents, 1 shared specialist.

    - 2 laborers (one per chain's collection)
    - 1 specialist (THE bottleneck — must alternate sort_A/sort_B AND
      process_A/process_B across both chains)
    - 1 vendor (must alternate retail_A/retail_B)

    The specialist and vendor are each single points of failure for two
    parallel chains. Their task-switching strategy IS the coordination
    problem.
    """
    agent_class = {
        "greedy": GreedyAgent,
        "field": FieldAgent,
        "planner": PlannerAgent,
    }[mechanism.value if hasattr(mechanism, "value") else mechanism]

    agents = []
    for i in range(2):
        cap = AgentCapability(
            id=f"laborer_{i}", capabilities=frozenset(["labor"]),
            base_efficiency=0.88,
        )
        agents.append(FieldAgent(cap, chain) if agent_class is FieldAgent
                      else agent_class(cap))

    cap = AgentCapability(
        id="specialist_0", capabilities=frozenset(["sorting", "processing"]),
        base_efficiency=0.92,
    )
    agents.append(FieldAgent(cap, chain) if agent_class is FieldAgent
                  else agent_class(cap))

    cap = AgentCapability(
        id="vendor_0", capabilities=frozenset(["sales"]),
        base_efficiency=0.90,
    )
    agents.append(FieldAgent(cap, chain) if agent_class is FieldAgent
                  else agent_class(cap))

    return agents


def crossroads_injection(tick: int) -> list:
    """Alternating injection: chain A gets energy on even ticks, B on odd.

    This makes the coordination problem dynamic — the specialist can't
    just camp on one chain. It must follow the energy.
    """
    if tick % 2 == 0:
        return [FieldEvent(event_type=EventType.CAPITAL_INJECTED,
                           task_id="collect_A", amount=50)]
    else:
        return [FieldEvent(event_type=EventType.CAPITAL_INJECTED,
                           task_id="collect_B", amount=50)]


# ═══════════════════════════════════════════════════════════════════════
#  SCENARIO 2: CASCADE
# ═══════════════════════════════════════════════════════════════════════

def build_cascade_scenario() -> tuple[ValueChain, EnergyField]:
    """Deep 5-hop value chain with a branching mid-point.

    Topology:
        mine → refine → assemble ─┬→ package_A → ship_A
                                 └→ package_B → ship_B

    Binding constraint: the chain is 5 hops deep. Energy injected at "mine"
    takes 5 sequential completions to reach a terminal market. Greedy agents
    pile up at the front (mine/refine) because that's where energy is
    highest. But the chain STALLS at assemble because nobody pre-positions
    downstream.

    Field agents should recognize that completing "refine" pushes energy to
    "assemble", which then needs agents ready at package_A/B. They should
    spread along the chain rather than clumping at the source.

    The branching at "assemble" creates a second coordination problem:
    agents must balance two downstream paths.
    """
    chain = ValueChain("cascade")

    tasks = [
        Task(id="mine", task_type=TaskType.COLLECTION,
             energy_capacity=150.0,
             required_capabilities=frozenset(["labor"])),
        Task(id="refine", task_type=TaskType.PROCESSING,
             energy_capacity=80.0,
             required_capabilities=frozenset(["refining"])),
        Task(id="assemble", task_type=TaskType.PROCESSING,
             energy_capacity=60.0,
             required_capabilities=frozenset(["assembly"])),
        Task(id="package_A", task_type=TaskType.PROCESSING,
             energy_capacity=50.0,
             required_capabilities=frozenset(["packaging"])),
        Task(id="package_B", task_type=TaskType.PROCESSING,
             energy_capacity=50.0,
             required_capabilities=frozenset(["packaging"])),
        Task(id="ship_A", task_type=TaskType.DISTRIBUTION,
             energy_capacity=40.0,
             required_capabilities=frozenset(["logistics"])),
        Task(id="ship_B", task_type=TaskType.DISTRIBUTION,
             energy_capacity=40.0,
             required_capabilities=frozenset(["logistics"])),
    ]

    for t in tasks:
        chain.add_task(t)

    couplings = [
        Coupling("mine", "refine", coefficient=0.85),
        Coupling("refine", "assemble", coefficient=0.85),
        Coupling("assemble", "package_A", coefficient=0.5),
        Coupling("assemble", "package_B", coefficient=0.5),
        Coupling("package_A", "ship_A", coefficient=1.0),
        Coupling("package_B", "ship_B", coefficient=1.0),
    ]
    for c in couplings:
        chain.add_coupling(c)

    field = EnergyField(chain, initial_energy=600.0)
    return chain, field


CASCADE_POSITIONS = {
    "mine":      {"x": 0.06, "y": 0.50},
    "refine":    {"x": 0.24, "y": 0.50},
    "assemble":  {"x": 0.42, "y": 0.50},
    "package_A": {"x": 0.62, "y": 0.25},
    "package_B": {"x": 0.62, "y": 0.75},
    "ship_A":    {"x": 0.85, "y": 0.25},
    "ship_B":    {"x": 0.85, "y": 0.75},
}


def make_cascade_agents(mechanism, chain: ValueChain) -> list:
    """Cascade roster: 6 agents with specialized capabilities.

    Each stage needs a different capability, so agents can't substitute
    across stages. The coordination problem: energy flows through the chain
    one hop at a time, and each hop needs a different specialist.
    """
    agent_class = {
        "greedy": GreedyAgent,
        "field": FieldAgent,
        "planner": PlannerAgent,
    }[mechanism.value if hasattr(mechanism, "value") else mechanism]

    roster = [
        ("worker_0", frozenset(["labor"]), 0.88),
        ("worker_1", frozenset(["labor"]), 0.85),
        ("refiner_0", frozenset(["refining"]), 0.90),
        ("assembler_0", frozenset(["assembly"]), 0.92),
        ("packer_0", frozenset(["packaging"]), 0.89),
        ("shipper_0", frozenset(["logistics"]), 0.87),
    ]

    agents = []
    for id_str, caps, eff in roster:
        cap = AgentCapability(id=id_str, capabilities=caps, base_efficiency=eff)
        agents.append(FieldAgent(cap, chain) if agent_class is FieldAgent
                      else agent_class(cap))
    return agents


def cascade_injection(tick: int) -> list:
    """Steady injection at the mine (source of the chain)."""
    if tick % 2 == 0:
        return [FieldEvent(event_type=EventType.CAPITAL_INJECTED,
                           task_id="mine", amount=60)]
    return []


# ═══════════════════════════════════════════════════════════════════════
#  SCENARIO 3: BURST
# ═══════════════════════════════════════════════════════════════════════

def build_burst_scenario() -> tuple[ValueChain, EnergyField]:
    """Temporal pressure: burst injection creates wave dynamics.

    Topology:
        supply_north ─┐
        supply_south ─┼→ warehouse → distribute → customer
        supply_east  ─┘

    Binding constraint: energy arrives in BURSTS at random supply nodes
    (every 10 ticks, a large dump hits one node). The warehouse has tight
    capacity (40). Greedy agents all rush whichever supply node just got
    the burst, flooding the warehouse and causing overflow waste. Field
    agents should spread: some handle the burst, others keep the downstream
    (warehouse → distribute → customer) flowing so the warehouse doesn't
    overflow.

    This tests temporal coordination: reacting to bursts vs. maintaining
    steady downstream flow.
    """
    chain = ValueChain("burst")

    tasks = [
        Task(id="supply_north", task_type=TaskType.COLLECTION,
             energy_capacity=100.0,
             required_capabilities=frozenset(["labor"])),
        Task(id="supply_south", task_type=TaskType.COLLECTION,
             energy_capacity=100.0,
             required_capabilities=frozenset(["labor"])),
        Task(id="supply_east", task_type=TaskType.COLLECTION,
             energy_capacity=100.0,
             required_capabilities=frozenset(["labor"])),

        # Tight bottleneck
        Task(id="warehouse", task_type=TaskType.PROCESSING,
             energy_capacity=40.0,   # VERY tight — overflow is the threat
             required_capabilities=frozenset(["management"])),

        Task(id="distribute", task_type=TaskType.DISTRIBUTION,
             energy_capacity=60.0,
             required_capabilities=frozenset(["logistics"])),

        Task(id="customer", task_type=TaskType.RETAIL,
             energy_capacity=80.0,
             required_capabilities=frozenset(["sales"])),
    ]

    for t in tasks:
        chain.add_task(t)

    couplings = [
        Coupling("supply_north", "warehouse", coefficient=0.8),
        Coupling("supply_south", "warehouse", coefficient=0.8),
        Coupling("supply_east", "warehouse", coefficient=0.8),
        Coupling("warehouse", "distribute", coefficient=0.9),
        Coupling("distribute", "customer", coefficient=1.0),
    ]
    for c in couplings:
        chain.add_coupling(c)

    field = EnergyField(chain, initial_energy=400.0)
    return chain, field


BURST_POSITIONS = {
    "supply_north": {"x": 0.08, "y": 0.15},
    "supply_south": {"x": 0.08, "y": 0.50},
    "supply_east":  {"x": 0.08, "y": 0.85},
    "warehouse":    {"x": 0.35, "y": 0.50},
    "distribute":   {"x": 0.65, "y": 0.50},
    "customer":     {"x": 0.90, "y": 0.50},
}


def make_burst_agents(mechanism, chain: ValueChain) -> list:
    """Burst roster: 6 agents with mixed capabilities.

    - 3 laborers (can work any supply node)
    - 1 manager (THE warehouse bottleneck — only 1 agent can manage)
    - 1 logistics worker (distribute)
    - 1 sales agent (customer)

    The manager is the hard bottleneck: only 1 agent, tight capacity (40).
    If laborers flood supply nodes while the warehouse is full, energy
    overflows and is wasted. Field agents should recognize when the
    warehouse is near capacity and hold off on collection.
    """
    agent_class = {
        "greedy": GreedyAgent,
        "field": FieldAgent,
        "planner": PlannerAgent,
    }[mechanism.value if hasattr(mechanism, "value") else mechanism]

    roster = [
        ("laborer_0", frozenset(["labor"]), 0.87),
        ("laborer_1", frozenset(["labor"]), 0.85),
        ("laborer_2", frozenset(["labor"]), 0.83),
        ("manager_0", frozenset(["management"]), 0.93),
        ("logistics_0", frozenset(["logistics"]), 0.90),
        ("sales_0", frozenset(["sales"]), 0.88),
    ]

    agents = []
    for id_str, caps, eff in roster:
        cap = AgentCapability(id=id_str, capabilities=caps, base_efficiency=eff)
        agents.append(FieldAgent(cap, chain) if agent_class is FieldAgent
                      else agent_class(cap))
    return agents


import random as _random

_burst_rng = _random.Random(42)

def burst_injection(tick: int) -> list:
    """Burst injection: large energy dump at a random supply node every 10 ticks."""
    if tick % 10 == 0 and tick > 0:
        node = _burst_rng.choice(["supply_north", "supply_south", "supply_east"])
        return [FieldEvent(event_type=EventType.CAPITAL_INJECTED,
                           task_id=node, amount=120)]
    return []


# ═══════════════════════════════════════════════════════════════════════
#  ORIGINAL SCENARIO (trash-cleanup — kept for backward compat)
# ═══════════════════════════════════════════════════════════════════════

def build_trash_cleanup_scenario() -> tuple[ValueChain, EnergyField]:
    """Trash cleanup value chain — the canonical Synchronicity example."""
    chain = ValueChain("trash-cleanup")

    tasks = [
        Task(id="collect_north", task_type=TaskType.COLLECTION,
             location="north", energy_capacity=200.0,
             required_capabilities=frozenset(["labor"])),
        Task(id="collect_south", task_type=TaskType.COLLECTION,
             location="south", energy_capacity=200.0,
             required_capabilities=frozenset(["labor"])),
        Task(id="park_cleanup", task_type=TaskType.SERVICE,
             location="central_park", energy_capacity=150.0,
             required_capabilities=frozenset(["labor"])),
        Task(id="sort_center", task_type=TaskType.PROCESSING,
             location="center", energy_capacity=120.0,
             required_capabilities=frozenset(["sorting"])),
        Task(id="compost", task_type=TaskType.PROCESSING,
             location="center", energy_capacity=100.0,
             required_capabilities=frozenset(["processing"])),
        Task(id="recycling", task_type=TaskType.PROCESSING,
             location="east", energy_capacity=100.0,
             required_capabilities=frozenset(["processing"])),
        Task(id="market_stall", task_type=TaskType.RETAIL,
             location="downtown", energy_capacity=60.0,
             required_capabilities=frozenset(["sales"])),
        Task(id="wholesale", task_type=TaskType.DISTRIBUTION,
             location="port", energy_capacity=60.0,
             required_capabilities=frozenset(["sales"])),
    ]
    for t in tasks:
        chain.add_task(t)

    couplings = [
        Coupling("collect_north", "sort_center", coefficient=0.8),
        Coupling("collect_south", "sort_center", coefficient=0.8),
        Coupling("park_cleanup", "sort_center", coefficient=0.6),
        Coupling("sort_center", "compost", coefficient=0.5),
        Coupling("sort_center", "recycling", coefficient=0.5),
        Coupling("compost", "market_stall", coefficient=1.0),
        Coupling("recycling", "wholesale", coefficient=1.0),
    ]
    for c in couplings:
        chain.add_coupling(c)

    field = EnergyField(chain, initial_energy=1200.0)
    return chain, field


def make_agents_for_mechanism(mechanism, chain: ValueChain) -> list:
    """Agent roster for the trash-cleanup scenario."""
    agent_class = {
        "greedy": GreedyAgent,
        "field": FieldAgent,
        "planner": PlannerAgent,
    }[mechanism.value if hasattr(mechanism, "value") else mechanism]

    agents = []
    for i in range(3):
        cap = AgentCapability(
            id=f"laborer_{i}", capabilities=frozenset(["labor"]),
            base_efficiency=0.85 + (i * 0.03),
        )
        agents.append(FieldAgent(cap, chain) if agent_class is FieldAgent
                      else agent_class(cap))
    for i in range(2):
        cap = AgentCapability(
            id=f"specialist_{i}", capabilities=frozenset(["sorting", "processing"]),
            base_efficiency=0.90 + (i * 0.02),
        )
        agents.append(FieldAgent(cap, chain) if agent_class is FieldAgent
                      else agent_class(cap))
    cap = AgentCapability(
        id="vendor_0", capabilities=frozenset(["sales"]),
        base_efficiency=0.92,
    )
    agents.append(FieldAgent(cap, chain) if agent_class is FieldAgent
                  else agent_class(cap))
    return agents


# ═══════════════════════════════════════════════════════════════════════
#  REGISTRY (must be after all function definitions)
# ═══════════════════════════════════════════════════════════════════════

SCENARIOS = {
    "trash-cleanup": {
        "build": build_trash_cleanup_scenario,
        "agents": make_agents_for_mechanism,
        "positions": {
            "collect_north": {"x": 0.08, "y": 0.20},
            "collect_south": {"x": 0.08, "y": 0.50},
            "park_cleanup": {"x": 0.08, "y": 0.80},
            "sort_center": {"x": 0.32, "y": 0.50},
            "compost": {"x": 0.56, "y": 0.25},
            "recycling": {"x": 0.56, "y": 0.75},
            "market_stall": {"x": 0.82, "y": 0.25},
            "wholesale": {"x": 0.82, "y": 0.75},
        },
        "injection": lambda tick: (
            [
                FieldEvent(event_type=EventType.CAPITAL_INJECTED, task_id="collect_north", amount=80),
                FieldEvent(event_type=EventType.CAPITAL_INJECTED, task_id="collect_south", amount=80),
                FieldEvent(event_type=EventType.CAPITAL_INJECTED, task_id="park_cleanup", amount=50),
            ] if tick % 3 == 0 else []
        ),
        "ticks": 200,
        "batch_size": 3,
    },
    "crossroads": {
        "build": build_crossroads_scenario,
        "agents": make_crossroads_agents,
        "positions": CROSSROADS_POSITIONS,
        "injection": crossroads_injection,
        "ticks": 200,
        "batch_size": 2,
    },
    "cascade": {
        "build": build_cascade_scenario,
        "agents": make_cascade_agents,
        "positions": CASCADE_POSITIONS,
        "injection": cascade_injection,
        "ticks": 200,
        "batch_size": 3,
    },
    "burst": {
        "build": build_burst_scenario,
        "agents": make_burst_agents,
        "positions": BURST_POSITIONS,
        "injection": burst_injection,
        "ticks": 200,
        "batch_size": 3,
    },
}
