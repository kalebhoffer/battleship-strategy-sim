"""Exact counts of legal fleet layouts, by two independent methods.

`count_layouts` is a dynamic program over the cells in reading order and is fast enough
for the full 10x10 game. `count_layouts_brute` enumerates every layout one by one and is
only practical on small grids, where it serves as a check on the dynamic program.

Ships are labeled: swapping the Destroyer and the Submarine gives a different layout.
"""

from __future__ import annotations

import functools

from battleship_sim.board import fits, placements
from battleship_sim.rules import Rules


def count_layouts(rules: Rules) -> int:
    """Number of legal layouts of the whole fleet."""
    n = rules.size
    lengths = tuple(ship.length for ship in rules.fleet)
    full = (1 << len(lengths)) - 1
    window = (1 << (n + 1)) - 1  # how far back a ship's neighbors can reach

    def ship_cells(p: int, length: int, vertical: bool) -> tuple[int, ...] | None:
        row, col = divmod(p, n)
        if vertical:
            return None if row + length > n else tuple(p + k * n for k in range(length))
        return None if col + length > n else tuple(range(p, p + length))

    @functools.cache
    def neighbors(cells: tuple[int, ...]) -> tuple[int, ...]:
        own = set(cells)
        out = set()
        for cell in cells:
            row, col = divmod(cell, n)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    r, c = row + dr, col + dc
                    if 0 <= r < n and 0 <= c < n and r * n + c not in own:
                        out.add(r * n + c)
        return tuple(sorted(out))

    @functools.cache
    def walk(p: int, claims: int, history: int, used: int) -> int:
        # claims: bit k = cell p+k already belongs to a ship that started earlier.
        # history: bit j = cell p-1-j is occupied (tracked only when ships may not touch).
        if p == n * n:
            return int(used == full and claims == 0)
        if claims & 1:
            return walk(p + 1, claims >> 1, _push(history, 1, window, rules), used)
        total = walk(p + 1, claims >> 1, _push(history, 0, window, rules), used)
        for i, length in enumerate(lengths):
            if used >> i & 1:
                continue
            for vertical in (False, True) if length > 1 else (False,):
                cells = ship_cells(p, length, vertical)
                if cells is None:
                    continue
                mask = 0
                for cell in cells:
                    mask |= 1 << (cell - p)
                if claims & mask:
                    continue
                if not rules.touching and _touches(p, neighbors(cells), claims, history):
                    continue
                new = claims | mask
                total += walk(p + 1, new >> 1, _push(history, 1, window, rules), used | 1 << i)
        return total

    result = walk(0, 0, 0, 0)
    walk.cache_clear()
    return result


def _push(history: int, bit: int, window: int, rules: Rules) -> int:
    if rules.touching:
        return 0
    return ((history << 1) | bit) & window


def _touches(p: int, around: tuple[int, ...], claims: int, history: int) -> bool:
    for q in around:
        if q < p:
            if history >> (p - q - 1) & 1:
                return True
        elif claims >> (q - p) & 1:
            return True
    return False


def count_layouts_brute(rules: Rules) -> int:
    """Number of legal layouts, by listing them all. Small grids only."""
    options = [placements(ship.length, rules.size) for ship in rules.fleet]

    def place(i: int, taken: int) -> int:
        if i == len(options):
            return 1
        return sum(place(i + 1, taken | p) for p in options[i] if fits(p, taken, rules))

    return place(0, 0)


def enumerate_layouts(rules: Rules) -> list[tuple[int, ...]]:
    """Every legal layout. Small grids only."""
    options = [placements(ship.length, rules.size) for ship in rules.fleet]
    out: list[tuple[int, ...]] = []

    def place(prefix: tuple[int, ...], taken: int) -> None:
        if len(prefix) == len(options):
            out.append(prefix)
            return
        for p in options[len(prefix)]:
            if fits(p, taken, rules):
                place(prefix + (p,), taken | p)

    place((), 0)
    return out


def independent_positions(rules: Rules) -> int:
    """Number of ways to put each ship somewhere on the grid, ignoring the other ships."""
    total = 1
    for ship in rules.fleet:
        total *= len(placements(ship.length, rules.size))
    return total
