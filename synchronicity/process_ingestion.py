"""
Process-log ingestion: auto-discover value chains from business event logs.

This module bridges the gap between real business data and Synchronicity's
energy field model. Instead of hand-coding value chain topologies, we ingest
event logs (the format your Bodhi system was designed around) and auto-discover:

  1. Tasks (from event log activities)
  2. Couplings (from directly-follows relationships in the process model)
  3. Capability requirements (from resource/role assignments in the log)
  4. Energy capacities (from frequency and timing statistics)

This creates the "massive associative space" — a value chain topology
grounded in real business operations, potentially hundreds of tasks with
complex real-world dependencies.

Uses PM4Py (Fraunhofer FIT) — the exact tool referenced in the original
Synchronicity/Bodhi grant proposal.

Event Log Format (from the Synchronicity paper):
    Order#  Activity         Employee  Date        Time
    1338    Take Order       Lucy      2020-04-01  13:37
    1338    Register Payment Lucy      2020-04-01  13:40
    1338    Prepare Burger   Luigi     2020-04-01  13:41
    1338    Deliver Order    Mike      2020-04-01  13:55
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from synchronicity.field import EnergyField
from synchronicity.value_chain import ValueChain, Task, Coupling, TaskType

logger = logging.getLogger(__name__)

# Map common business activities to Synchronicity task types
_ACTIVITY_TYPE_MAP = {
    # Collection / intake
    "take": TaskType.COLLECTION, "order": TaskType.COLLECTION,
    "receive": TaskType.COLLECTION, "collect": TaskType.COLLECTION,
    "intake": TaskType.COLLECTION, "register": TaskType.COLLECTION,
    # Processing
    "process": TaskType.PROCESSING, "prepare": TaskType.PROCESSING,
    "sort": TaskType.PROCESSING, "review": TaskType.PROCESSING,
    "approve": TaskType.PROCESSING, "verify": TaskType.PROCESSING,
    "check": TaskType.PROCESSING, "validate": TaskType.PROCESSING,
    "analyze": TaskType.PROCESSING, "assess": TaskType.PROCESSING,
    # Service
    "clean": TaskType.SERVICE, "fix": TaskType.SERVICE,
    "install": TaskType.SERVICE, "maintain": TaskType.SERVICE,
    # Distribution
    "ship": TaskType.DISTRIBUTION, "deliver": TaskType.DISTRIBUTION,
    "send": TaskType.DISTRIBUTION, "transfer": TaskType.DISTRIBUTION,
    "route": TaskType.DISTRIBUTION, "dispatch": TaskType.DISTRIBUTION,
    # Retail
    "sell": TaskType.RETAIL, "invoice": TaskType.RETAIL,
    "charge": TaskType.RETAIL, "payment": TaskType.RETAIL,
    "checkout": TaskType.RETAIL,
}

def _infer_task_type(activity: str) -> TaskType:
    """Infer task type from activity name using keyword matching."""
    activity_lower = activity.lower()
    for keyword, task_type in _ACTIVITY_TYPE_MAP.items():
        if keyword in activity_lower:
            return task_type
    return TaskType.SERVICE  # default


def discover_value_chain(
    df: pd.DataFrame,
    case_col: str = "case:concept:name",
    activity_col: str = "concept:name",
    resource_col: str = "org:resource",
    timestamp_col: str = "time:timestamp",
    name: str = "discovered",
    capacity_multiplier: float = 3.0,
) -> tuple[ValueChain, dict[str, set[str]], dict[str, dict]]:
    """Discover a value chain from an event log.

    This is the core function that bridges process mining and Synchronicity.
    It takes a PM4Py-format event log and produces:
      - A ValueChain topology (tasks + couplings)
      - Capability assignments (resource → capabilities)
      - Activity statistics (frequency, avg duration, resource set)

    The directly-follows graph from process mining becomes the coupling
    structure: if activity A is followed by activity B in the event log,
    then completing A propagates energy to B.

    Args:
        df: Event log as a pandas DataFrame (PM4Py format).
        capacity_multiplier: Scale factor for energy capacities. Higher = more
            slack. Capacities are derived from activity frequency — activities
            that occur more often get higher capacity.

    Returns:
        (chain, task_resources, stats)
        - chain: ValueChain ready for simulation
        - task_resources: task_id → set of resource names that can do it
        - stats: task_id → {frequency, avg_duration, resources, ...}
    """
    # Ensure timestamp is datetime
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values([case_col, timestamp_col])

    activities = df[activity_col].unique()

    # ── Activity statistics ─────────────────────────────────────
    activity_stats: dict[str, dict] = {}
    for activity in activities:
        activity_df = df[df[activity_col] == activity]
        frequency = len(activity_df)
        resources = set(activity_df[resource_col].dropna().unique()) if resource_col in df.columns else set()

        # Average duration: time between this activity and the next in each case
        durations = []
        for case_id in activity_df[case_col].unique():
            case_events = df[df[case_col] == case_id].sort_values(timestamp_col)
            act_indices = case_events[case_events[activity_col] == activity].index
            for idx in act_indices:
                next_events = case_events.loc[idx:]
                if len(next_events) > 1:
                    duration = next_events.iloc[1][timestamp_col] - next_events.iloc[0][timestamp_col]
                    durations.append(duration.total_seconds())

        avg_duration = sum(durations) / len(durations) if durations else 3600  # default 1 hour

        activity_stats[activity] = {
            "frequency": frequency,
            "avg_duration_seconds": avg_duration,
            "resources": resources,
            "task_type": _infer_task_type(activity),
        }

    # ── Build value chain ───────────────────────────────────────
    chain = ValueChain(name)

    # Sanitize activity names for use as task IDs
    def sanitize(name: str) -> str:
        return name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")

    task_id_map: dict[str, str] = {}
    for activity in activities:
        task_id = sanitize(activity)
        # Ensure uniqueness
        if task_id in task_id_map.values():
            task_id = f"{task_id}_{activities.tolist().index(activity)}"
        task_id_map[activity] = task_id

        stats = activity_stats[activity]
        # Energy capacity proportional to frequency (more common = more capacity)
        capacity = max(20.0, min(stats["frequency"] * capacity_multiplier, 300.0))

        # Required capabilities = the resource names (each resource is a "capability")
        # In a real deployment this would be roles/skills, not individual people
        caps = frozenset(sanitize(r) for r in stats["resources"]) if stats["resources"] else frozenset()

        chain.add_task(Task(
            id=task_id,
            task_type=stats["task_type"],
            location="discovered",
            energy_capacity=capacity,
            required_capabilities=caps,
        ))

    # ── Discover couplings from directly-follows graph ──────────
    # If A → B appears in the event log (A directly followed by B in same case),
    # then A couples to B. Coupling strength = frequency of the A→B transition
    # relative to all outgoing transitions from A.
    follows_count: dict[str, Counter] = defaultdict(Counter)
    for case_id in df[case_col].unique():
        case_df = df[df[case_col] == case_col].sort_values(timestamp_col)
        case_df = df[df[case_col] == case_id].sort_values(timestamp_col)
        activities_in_case = case_df[activity_col].tolist()
        for i in range(len(activities_in_case) - 1):
            a = activities_in_case[i]
            b = activities_in_case[i + 1]
            follows_count[a][b] += 1

    for source_activity, targets in follows_count.items():
        source_id = task_id_map[source_activity]
        total = sum(targets.values())
        for target_activity, count in targets.items():
            target_id = task_id_map[target_activity]
            coefficient = min(count / total, 1.0)
            try:
                chain.add_coupling(Coupling(
                    source=source_id,
                    target=target_id,
                    coefficient=coefficient,
                    description=f"{source_activity} → {target_activity} ({count} occurrences)",
                ))
            except ValueError as e:
                logger.debug(f"Skipping coupling {source_id}→{target_id}: {e}")

    # ── Build resource → capabilities map ───────────────────────
    task_resources: dict[str, set[str]] = {}
    for activity, task_id in task_id_map.items():
        task_resources[task_id] = activity_stats[activity]["resources"]

    logger.info(
        f"Discovered value chain '{name}': {len(chain)} tasks, "
        f"{len(chain.graph.edges)} couplings from {len(df)} events"
    )

    return chain, task_resources, activity_stats


def generate_synthetic_event_log(
    n_cases: int = 100,
    n_variants: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic business event log for testing.

    Creates a realistic order-fulfillment process with variants:
    - Standard: order → verify → process → ship → invoice
    - Express:  order → process → ship → invoice
    - Returns:  order → verify → process → return → refund

    This gives us a multi-variant process model (like real businesses)
    with branching and merging — exactly the kind of topology where
    coordination strategy matters.
    """
    rng = random.Random(seed)

    variants = [
        ["Take Order", "Verify Payment", "Prepare Item", "Quality Check",
         "Ship Item", "Send Invoice"],
        ["Take Order", "Prepare Item", "Ship Item", "Send Invoice"],
        ["Take Order", "Verify Payment", "Prepare Item", "Quality Check",
         "Return Item", "Process Refund"],
    ]

    activities_pool = ["Take Order", "Verify Payment", "Prepare Item",
                       "Quality Check", "Ship Item", "Send Invoice",
                       "Return Item", "Process Refund"]
    resources = ["Alice", "Bob", "Carol", "David", "Eve"]

    # Assign resources to activities (capability constraints)
    activity_resources = {
        "Take Order": {"Alice", "Bob"},
        "Verify Payment": {"Carol"},
        "Prepare Item": {"David", "Eve"},
        "Quality Check": {"Carol", "David"},
        "Ship Item": {"Bob", "Eve"},
        "Send Invoice": {"Alice"},
        "Return Item": {"Bob", "David"},
        "Process Refund": {"Carol", "Alice"},
    }

    rows = []
    for case_idx in range(n_cases):
        case_id = f"CASE_{case_idx:04d}"
        variant = rng.choice(variants[:n_variants])
        base_time = datetime(2024, 1, 1, 9, 0) + timedelta(
            days=rng.randint(0, 90),
            hours=rng.randint(0, 8),
        )

        for step_idx, activity in enumerate(variant):
            resource = rng.choice(list(activity_resources.get(activity, resources)))
            timestamp = base_time + timedelta(
                hours=step_idx * rng.uniform(1, 8),
                minutes=rng.randint(0, 59),
            )
            rows.append({
                "case:concept:name": case_id,
                "concept:name": activity,
                "org:resource": resource,
                "time:timestamp": timestamp,
            })

    df = pd.DataFrame(rows)
    logger.info(f"Generated synthetic log: {len(df)} events, {n_cases} cases")
    return df


def make_event_log_agents(
    chain: ValueChain,
    task_resources: dict[str, set[str]],
    mechanism,
    n_agents: int = 6,
) -> list:
    """Create agents from discovered task resources.

    Maps real resources (people/roles from the event log) to agents.
    Each agent's capabilities are derived from which tasks that resource
    performed in the event log.
    """
    from synchronicity.agents import (
        AgentCapability, GreedyAgent, FieldAgent, PlannerAgent,
    )

    agent_class = {
        "greedy": GreedyAgent,
        "field": FieldAgent,
        "planner": PlannerAgent,
    }[mechanism.value if hasattr(mechanism, "value") else mechanism]

    # Collect all unique resources and their capabilities
    resource_caps: dict[str, set[str]] = defaultdict(set)
    for task_id, resources in task_resources.items():
        for resource in resources:
            # The resource's capabilities = tasks they can do
            sanitized = resource.lower().replace(" ", "_")
            resource_caps[sanitized].add(task_id)

    # Hmm — we need capabilities as task-level requirements, not the other way.
    # Actually the chain tasks already have required_capabilities set from discovery.
    # Let's just use the resource names as capabilities.
    # Rebuild: each task's required_capabilities = set of resource names
    # Each agent = one resource, capabilities = resource name

    all_resources = set()
    for task_id, task in chain.tasks.items():
        all_resources.update(task.required_capabilities)

    if not all_resources:
        # Fallback: create generic agents
        all_resources = {f"agent_{i}" for i in range(n_agents)}

    agents = []
    resource_list = sorted(all_resources)

    for i, resource in enumerate(resource_list[:n_agents]):
        cap = AgentCapability(
            id=resource,
            capabilities=frozenset([resource]),
            base_efficiency=0.85 + (i % 3) * 0.03,
        )
        if agent_class is FieldAgent:
            agents.append(FieldAgent(cap, chain))
        else:
            agents.append(agent_class(cap))

    # If we have fewer resources than n_agents, pad with cross-trained agents
    while len(agents) < n_agents:
        # Give them multiple capabilities
        caps = set(rng.choice(resource_list, min(2, len(resource_list)))
                   for _ in range(2)) if resource_list else set()
        if not caps:
            caps = {f"generalist_{len(agents)}"}
        cap = AgentCapability(
            id=f"generalist_{len(agents)}",
            capabilities=frozenset(caps),
            base_efficiency=0.82,
        )
        if agent_class is FieldAgent:
            agents.append(FieldAgent(cap, chain))
        else:
            agents.append(agent_class(cap))

    return agents


def event_log_injection_schedule(
    chain: ValueChain,
    stats: dict[str, dict],
    interval: int = 3,
    amount: float = 60.0,
):
    """Create an injection schedule based on the discovered process.

    Injects energy into source tasks (tasks with no upstream couplings) at
    a rate proportional to their observed frequency in the event log.
    """
    import random as _r
    rng = _r.Random(42)

    # Find source tasks (no incoming edges)
    source_ids = [
        tid for tid in chain.all_task_ids()
        if not chain.upstream(tid)
    ]

    if not source_ids:
        source_ids = chain.all_task_ids()[:3]

    max_freq = max(
        (stats.get(chain.tasks[tid].id, {}).get("frequency", 1)
         for tid in source_ids),
        default=1,
    )

    # Normalize frequencies to get injection weights
    weights = []
    for tid in source_ids:
        task = chain.tasks[tid]
        freq = stats.get(tid, {}).get("frequency", 1) if stats else 1
        weights.append(freq / max_freq)

    def schedule(tick: int) -> list:
        from synchronicity.reward_machine import FieldEvent, EventType
        if tick % interval == 0:
            events = []
            for tid, w in zip(source_ids, weights):
                events.append(FieldEvent(
                    event_type=EventType.CAPITAL_INJECTED,
                    task_id=tid,
                    amount=amount * w,
                ))
            return events
        return []

    return schedule
