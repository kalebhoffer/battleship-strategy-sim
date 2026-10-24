"""Ship placement strategies. A placer turns the rules and an RNG into a legal layout."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from battleship_sim.board import Layout
from battleship_sim.placers.basic import naive_layout, uniform_layout
from battleship_sim.placers.styles import (
    anti_density_layout,
    center_layout,
    clustered_layout,
    edges_layout,
    mixed_placer,
    spread_layout,
)
from battleship_sim.rules import Rules

Placer = Callable[[Rules, random.Random], Layout]


@dataclass(frozen=True)
class PlacerSpec:
    name: str
    place: Placer
    description: str


PLACERS: dict[str, PlacerSpec] = {
    spec.name: spec
    for spec in (
        PlacerSpec(
            "uniform",
            uniform_layout,
            "Every legal fleet layout equally likely. The neutral baseline.",
        ),
        PlacerSpec(
            "naive",
            naive_layout,
            "Each ship at a random spot that still fits, largest first. Not uniform.",
        ),
        PlacerSpec(
            "edges",
            edges_layout,
            "Every ship touches the outer edge of the grid.",
        ),
        PlacerSpec(
            "center",
            center_layout,
            "Every ship inside the middle 6x6 of the grid.",
        ),
        PlacerSpec(
            "clustered",
            clustered_layout,
            "All ships in one group, each touching another side to side.",
        ),
        PlacerSpec(
            "spread",
            spread_layout,
            "The most spread out of 50 random layouts.",
        ),
        PlacerSpec(
            "anti-density",
            anti_density_layout,
            "Each ship where the probability map expects ships least, before any shot.",
        ),
    )
}


def get_placer(name: str) -> Placer:
    """A placer by name, or a mix such as 'mix:edges=0.3,uniform=0.7'."""
    if name.startswith("mix:"):
        return mixed_placer(name[4:], get_placer)
    try:
        return PLACERS[name].place
    except KeyError:
        raise ValueError(
            f"unknown placer {name!r}; choose from {', '.join(PLACERS)} or mix:name=weight,..."
        ) from None
