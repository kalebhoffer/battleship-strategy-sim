"""Shooting strategies.

Every shooter is built by a factory that receives only the rules and an RNG, never the
layout, and implements `fire()` and `observe(cell, result)` (see game.Shooter).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from battleship_sim.game import Shooter
from battleship_sim.rules import Rules
from battleship_sim.shooters.hunt import (
    adaptive_parity,
    hunt_target,
    hunt_target_parity,
    pattern,
)
from battleship_sim.shooters.naive import RandomRepeatShooter, RandomShooter
from battleship_sim.shooters.probability import ProbabilityMap

ShooterFactory = Callable[[Rules, random.Random], Shooter]


@dataclass(frozen=True)
class ShooterSpec:
    name: str
    factory: ShooterFactory
    description: str


SHOOTERS: dict[str, ShooterSpec] = {
    spec.name: spec
    for spec in (
        ShooterSpec(
            "random-repeat",
            RandomRepeatShooter,
            "Any cell at random, even ones already fired at. The floor.",
        ),
        ShooterSpec(
            "random",
            RandomShooter,
            "A random cell it has not fired at yet. The baseline.",
        ),
        ShooterSpec(
            "hunt-target",
            hunt_target,
            "Random search; after a hit, fires at the neighbors until none are left.",
        ),
        ShooterSpec(
            "hunt-target-parity",
            hunt_target_parity,
            "Hunt and target, searching only one color of a checkerboard.",
        ),
        ShooterSpec(
            "adaptive-parity",
            adaptive_parity,
            "Hunt and target on stripes spaced by the shortest ship still afloat.",
        ),
        ShooterSpec(
            "pattern-rows",
            pattern("rows"),
            "Hunt and target, searching A1, A2, ... row by row.",
        ),
        ShooterSpec(
            "pattern-diagonal",
            pattern("diagonal"),
            "Hunt and target, sweeping every other diagonal from the top left.",
        ),
        ShooterSpec(
            "pattern-spiral-in",
            pattern("spiral-in"),
            "Hunt and target, spiraling from the edges toward the center.",
        ),
        ShooterSpec(
            "pattern-spiral-out",
            pattern("spiral-out"),
            "Hunt and target, spiraling from the center out to the edges.",
        ),
        ShooterSpec(
            "probability-map",
            ProbabilityMap,
            "Fires where ships are most likely to be, counting every position that fits.",
        ),
    )
}


def make_shooter(name: str, rules: Rules, rng: random.Random) -> Shooter:
    try:
        spec = SHOOTERS[name]
    except KeyError:
        raise ValueError(f"unknown shooter {name!r}; choose from {', '.join(SHOOTERS)}") from None
    return spec.factory(rules, rng)
