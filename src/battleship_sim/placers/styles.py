"""Placement styles: ways a person might choose to hide a fleet.

Styles that restrict each ship separately (edges, center, anti-density) are uniform over
every legal layout that meets the restriction. Clustered and spread are procedures, not
uniform over some set, and say so.
"""

from __future__ import annotations

import functools
import random

from battleship_sim.board import Layout, cells_of, fits, placements
from battleship_sim.placers.basic import MAX_ATTEMPTS, uniform_layout
from battleship_sim.rules import FLEET, Rules

SPREAD_CANDIDATES = 50


def _uniform_among(options: list[tuple[int, ...]], rules: Rules, rng: random.Random) -> Layout:
    """Uniform over legal layouts whose i-th ship is drawn from options[i]."""
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
    raise RuntimeError("no legal layout found under this placement style")


def _border(size: int) -> int:
    mask = 0
    for cell in range(size * size):
        row, col = divmod(cell, size)
        if row in (0, size - 1) or col in (0, size - 1):
            mask |= 1 << cell
    return mask


def edges_layout(rules: Rules, rng: random.Random) -> Layout:
    """Every ship touches the outer edge of the grid with at least one square."""
    border = _border(rules.size)
    options = [
        tuple(p for p in placements(s.length, rules.size) if p & border) for s in rules.fleet
    ]
    return _uniform_among(options, rules, rng)


def center_margin(size: int) -> int:
    return size // 5  # two rows and columns on a 10x10 grid, leaving the middle 6x6


def center_layout(rules: Rules, rng: random.Random) -> Layout:
    """Every ship lies entirely in the middle of the grid (the middle 6x6 on 10x10)."""
    n, m = rules.size, center_margin(rules.size)
    if not rules.touching and n == 10 and rules.fleet == FLEET:
        # count_layouts(Rules(size=6, touching=False)) == 0: the fleet can't fit.
        raise ValueError("the official fleet can't fit in the middle 6x6 without touching")
    inside = 0
    for cell in range(n * n):
        row, col = divmod(cell, n)
        if m <= row < n - m and m <= col < n - m:
            inside |= 1 << cell
    options = [
        tuple(p for p in placements(s.length, n) if p & inside == p) for s in rules.fleet
    ]
    if not all(options):
        raise ValueError("a ship is longer than the center region")
    return _uniform_among(options, rules, rng)


def _neighbors(mask: int, size: int) -> int:
    """Squares sharing a side with `mask`, excluding `mask` itself."""
    out = 0
    for cell in cells_of(mask):
        row, col = divmod(cell, size)
        for r, c in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= r < size and 0 <= c < size:
                out |= 1 << (r * size + c)
    return out & ~mask


def clustered_layout(rules: Rules, rng: random.Random) -> Layout:
    """All ships form one group, each touching another side to side.

    Ships go down in random order: the first anywhere, each later one at a random spot
    that touches the group. A procedure, not uniform over all clustered layouts.
    """
    if not rules.touching:
        raise ValueError("clustered placement needs ships to be allowed to touch")
    n = rules.size
    for _ in range(MAX_ATTEMPTS):
        order = list(range(len(rules.fleet)))
        rng.shuffle(order)
        layout: list[int] = [0] * len(order)
        taken = 0
        for k, i in enumerate(order):
            room = [
                p
                for p in placements(rules.fleet[i].length, n)
                if not p & taken and (k == 0 or p & _neighbors(taken, n))
            ]
            if not room:
                break
            layout[i] = rng.choice(room)
            taken |= layout[i]
        else:
            return tuple(layout)
    raise RuntimeError("no clustered layout found")


def _gap(a: int, b: int, size: int) -> int:
    """Smallest king-move distance between a square of ship a and a square of ship b."""
    best = size
    for x in cells_of(a):
        xr, xc = divmod(x, size)
        for y in cells_of(b):
            yr, yc = divmod(y, size)
            best = min(best, max(abs(xr - yr), abs(xc - yc)))
    return best


def spread_score(layout: Layout, size: int) -> tuple[int, int]:
    """(smallest gap between any two ships, total of all pairwise gaps)."""
    gaps = [
        _gap(a, b, size) for i, a in enumerate(layout) for b in layout[i + 1 :]
    ]
    return min(gaps), sum(gaps)


def spread_layout(rules: Rules, rng: random.Random) -> Layout:
    """The most spread out of 50 random layouts: largest smallest gap between two ships,
    then the largest total gap."""
    candidates = [uniform_layout(rules, rng) for _ in range(SPREAD_CANDIDATES)]
    return max(candidates, key=lambda layout: spread_score(layout, rules.size))


@functools.cache
def opening_odds(rules: Rules) -> tuple[float, ...]:
    """The probability map's score for every square before any shot is fired."""
    from battleship_sim.shooters.probability import ProbabilityMap

    return tuple(ProbabilityMap(rules, random.Random(0)).scores())


@functools.cache
def _quiet_options(rules: Rules) -> tuple[tuple[int, ...], ...]:
    odds = opening_odds(rules)
    options = []
    for ship in rules.fleet:
        ranked = sorted(
            placements(ship.length, rules.size),
            key=lambda p: sum(odds[c] for c in cells_of(p)),
        )
        options.append(tuple(ranked[: max(1, len(ranked) // 4)]))
    return tuple(options)


def anti_density_layout(rules: Rules, rng: random.Random) -> Layout:
    """Each ship only in the quarter of its positions the probability map rates least
    likely before the first shot (mostly edges and corners)."""
    return _uniform_among(list(_quiet_options(rules)), rules, rng)


def mixed_placer(spec: str, get):
    """'mix:edges=0.3,uniform=0.7' -> picks a style per game with those weights."""
    parts = []
    for item in spec.split(","):
        name, _, weight = item.partition("=")
        if not weight:
            raise ValueError(f"mix needs name=weight pairs, got {item!r}")
        parts.append((get(name.strip()), float(weight)))
    total = sum(w for _, w in parts)
    if total <= 0 or any(w < 0 for _, w in parts):
        raise ValueError("mix weights must be non-negative and not all zero")

    def place(rules: Rules, rng: random.Random) -> Layout:
        pick = rng.random() * total
        for placer, weight in parts:
            pick -= weight
            if pick < 0:
                return placer(rules, rng)
        return parts[-1][0](rules, rng)

    return place
