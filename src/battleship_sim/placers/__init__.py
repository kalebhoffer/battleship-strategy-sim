"""Ship placement strategies. A placer turns the rules and an RNG into a legal layout."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from battleship_sim.board import Layout
from battleship_sim.placers.basic import naive_layout, uniform_layout
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
    )
}


def get_placer(name: str) -> Placer:
    try:
        return PLACERS[name].place
    except KeyError:
        raise ValueError(f"unknown placer {name!r}; choose from {', '.join(PLACERS)}") from None
