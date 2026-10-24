"""Random placers: one exactly uniform over whole layouts, one the way people often code it."""

from __future__ import annotations

import random

from battleship_sim.board import Layout, fits, placements
from battleship_sim.rules import Rules

MAX_ATTEMPTS = 1_000_000


def uniform_layout(rules: Rules, rng: random.Random) -> Layout:
    """Uniform over all legal layouts, by rejection sampling.

    Each ship is drawn independently from all of its positions, which makes every tuple
    of positions equally likely; keeping only the legal tuples leaves every legal layout
    equally likely.
    """
    options = [placements(ship.length, rules.size) for ship in rules.fleet]
    for _ in range(MAX_ATTEMPTS):
        layout = []
        taken = 0
        for choices in options:
            placement = rng.choice(choices)
            if not fits(placement, taken, rules):
                break
            layout.append(placement)
            taken |= placement
        else:
            return tuple(layout)
    raise RuntimeError("no legal layout found; is the fleet too large for the grid?")


def naive_layout(rules: Rules, rng: random.Random) -> Layout:
    """Places ships one at a time, each uniformly among the spots that still fit.

    Ships go in fleet order (largest first in the official fleet). If a ship has no room
    left, the whole fleet starts over. Layouts that leave more room for later ships come
    up less often than under `uniform_layout`, so this is not uniform.
    """
    for _ in range(MAX_ATTEMPTS):
        layout = []
        taken = 0
        for ship in rules.fleet:
            room = [p for p in placements(ship.length, rules.size) if fits(p, taken, rules)]
            if not room:
                break
            placement = rng.choice(room)
            layout.append(placement)
            taken |= placement
        else:
            return tuple(layout)
    raise RuntimeError("no legal layout found; is the fleet too large for the grid?")
