"""
Value chain topology: task types and coupling structure.

This module defines the *static* topology of an economic system — what kinds
of tasks exist, how they connect to each other (value chains), and how
strongly completing one task creates value for the next (coupling coefficients).

This is the "geometry" of the economic space, analogous to the metric tensor
g_μν in general relativity. It describes the structure; the energy field
(described in field.py) describes what flows through that structure.

Coupling coefficients κ_ij encode the "synergy" described in the Synchronicity
thesis: completing task i creates MORE value than the task alone if it feeds
into task j, because the coupling means resources flow downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Optional

import networkx as nx


class TaskType(str, Enum):
    """Categories of economic activity in a value chain.

    Not prescriptive — systems define their own task types. This enum just
    provides a common vocabulary for the demo scenarios.
    """

    COLLECTION = "collection"      # gather raw materials / inputs
    PROCESSING = "processing"      # transform inputs into intermediate goods
    DISTRIBUTION = "distribution"  # move goods to where they're needed
    SERVICE = "service"            # provide a direct service to people
    RETAIL = "retail"              # sell to end consumers
    GOVERNANCE = "governance"      # coordinate / manage / oversee


@dataclass(frozen=True)
class Task:
    """A single economic task — a node in the value chain.

    Attributes:
        id: Unique identifier.
        task_type: Category of work.
        location: Abstract location tag (zone, region, etc.).
        energy_capacity: Maximum potential energy this task can hold.
            Acts as a "reservoir size" — prevents runaway accumulation.
        required_capabilities: Set of capability tags an agent needs to
            perform this task. Agents without matching capabilities suffer
            an efficiency penalty or cannot do the task at all.
    """

    id: str
    task_type: TaskType
    location: str = "default"
    energy_capacity: float = 100.0
    required_capabilities: frozenset[str] = frozenset()

    def __post_init__(self):
        if self.energy_capacity <= 0:
            raise ValueError(f"Task {self.id}: energy_capacity must be positive")


@dataclass
class Coupling:
    """A directed coupling between two tasks — an edge in the value chain.

    Represents the "synergy" effect: completing the source task injects energy
    into the target task, creating a value chain.

    Attributes:
        source: Source task id.
        target: Target task id.
        coefficient: κ ∈ (0, 1]. Fraction of propagating energy that flows
            along this edge when the source task completes. Multiple outgoing
            edges from the same node are normalized at field-update time.
        description: Human-readable note for what this coupling represents.
    """

    source: str
    target: str
    coefficient: float = 1.0
    description: str = ""


class ValueChain:
    """The static topology of an economic system.

    A directed graph of tasks connected by couplings. This is agent-agnostic
    and energy-agnostic — it describes the *shape* of the economy, not what's
    flowing through it.

    Analogous to the metric tensor in GR: it defines the geometry that
    determines how energy/matter (agents, capital) moves.
    """

    def __init__(self, name: str = "untitled"):
        self.name = name
        self._graph: nx.DiGraph = nx.DiGraph()

    def add_task(self, task: Task) -> None:
        """Add a task node to the value chain."""
        self._graph.add_node(
            task.id,
            task=task,
            task_type=task.task_type,
            location=task.location,
            energy_capacity=task.energy_capacity,
            required_capabilities=task.required_capabilities,
        )

    def add_coupling(self, coupling: Coupling) -> None:
        """Add a directed coupling (value chain edge) between two tasks.

        Raises:
            ValueError: if source or target task doesn't exist.
        """
        if coupling.source not in self._graph:
            raise ValueError(f"Coupling source '{coupling.source}' not in chain")
        if coupling.target not in self._graph:
            raise ValueError(f"Coupling target '{coupling.target}' not in chain")
        if not (0 < coupling.coefficient <= 1):
            raise ValueError(
                f"Coupling {coupling.source}→{coupling.target}: "
                f"coefficient must be in (0, 1], got {coupling.coefficient}"
            )

        self._graph.add_edge(
            coupling.source,
            coupling.target,
            coefficient=coupling.coefficient,
            description=coupling.description,
        )

    @property
    def tasks(self) -> dict[str, Task]:
        """All tasks in the chain, keyed by id."""
        return {nid: self._graph.nodes[nid]["task"] for nid in self._graph.nodes}

    def get_task(self, task_id: str) -> Task:
        return self._graph.nodes[task_id]["task"]

    def downstream(self, task_id: str) -> list[tuple[str, float]]:
        """Tasks that receive energy when this task completes.

        Returns:
            List of (target_id, coefficient) pairs for outgoing edges.
        """
        return [
            (target, data["coefficient"])
            for _, target, data in self._graph.out_edges(task_id, data=True)
        ]

    def upstream(self, task_id: str) -> list[tuple[str, float]]:
        """Tasks that feed energy into this task."""
        return [
            (source, data["coefficient"])
            for source, _, data in self._graph.in_edges(task_id, data=True)
        ]

    def downstream_transitive(self, task_id: str) -> set[str]:
        """All tasks reachable downstream (full value chain from this node)."""
        return nx.descendants(self._graph, task_id)

    def all_task_ids(self) -> list[str]:
        return list(self._graph.nodes)

    @property
    def graph(self) -> nx.DiGraph:
        """Direct access to the underlying networkx graph (for visualization)."""
        return self._graph

    def __len__(self) -> int:
        return len(self._graph)

    def summary(self) -> str:
        n_tasks = len(self._graph.nodes)
        n_edges = len(self._graph.edges)
        return (
            f"ValueChain('{self.name}'): {n_tasks} tasks, {n_edges} couplings"
        )
