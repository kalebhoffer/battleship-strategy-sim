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
from battleship_sim.shooters.naive import RandomRepeatShooter, RandomShooter

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
    )
}


def make_shooter(name: str, rules: Rules, rng: random.Random) -> Shooter:
    try:
        spec = SHOOTERS[name]
    except KeyError:
        raise ValueError(f"unknown shooter {name!r}; choose from {', '.join(SHOOTERS)}") from None
    return spec.factory(rules, rng)
