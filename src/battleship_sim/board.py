"""The grid, ship placements, and fleet layouts.

Cells are numbered row by row: cell = row * size + col, row 0 = A, col 0 = 1.
A set of cells is an int bitmask, so overlap checks are a single `&`.
A layout is a tuple with one placement mask per ship, in fleet order.
"""

from __future__ import annotations

import functools
import string

from battleship_sim.rules import Rules

Layout = tuple[int, ...]


def cell_name(cell: int, size: int) -> str:
    row, col = divmod(cell, size)
    return f"{string.ascii_uppercase[row]}{col + 1}"


def parse_cell(name: str, size: int) -> int:
    row = string.ascii_uppercase.index(name[0].upper())
    col = int(name[1:]) - 1
    if not (0 <= row < size and 0 <= col < size):
        raise ValueError(f"{name} is off a {size}x{size} grid")
    return row * size + col


def cells_of(mask: int) -> list[int]:
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


@functools.cache
def placements(length: int, size: int) -> tuple[int, ...]:
    """Every in-bounds position of one ship: horizontal ones first, then vertical.

    A one-cell ship has a single orientation, so it is not counted twice.
    """
    run = (1 << length) - 1
    horizontal = tuple(
        run << (r * size + c) for r in range(size) for c in range(size - length + 1)
    )
    if length == 1:
        return horizontal
    column = sum(1 << (k * size) for k in range(length))
    vertical = tuple(
        column << (r * size + c) for r in range(size - length + 1) for c in range(size)
    )
    return horizontal + vertical


@functools.cache
def halo(mask: int, size: int) -> int:
    """The cells of `mask` plus every cell touching them, diagonals included."""
    out = 0
    for cell in cells_of(mask):
        row, col = divmod(cell, size)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, c = row + dr, col + dc
                if 0 <= r < size and 0 <= c < size:
                    out |= 1 << (r * size + c)
    return out


def fits(placement: int, taken: int, rules: Rules) -> bool:
    """Can a ship go at `placement` given the cells `taken` by ships already placed?"""
    if rules.touching:
        return not placement & taken
    return not halo(placement, rules.size) & taken


def is_legal(layout: Layout, rules: Rules) -> bool:
    if len(layout) != len(rules.fleet):
        return False
    taken = 0
    for ship, placement in zip(rules.fleet, layout, strict=True):
        if placement not in _placement_set(ship.length, rules.size):
            return False
        if not fits(placement, taken, rules):
            return False
        taken |= placement
    return True


@functools.cache
def _placement_set(length: int, size: int) -> frozenset[int]:
    return frozenset(placements(length, size))


def occupied(layout: Layout) -> int:
    out = 0
    for placement in layout:
        out |= placement
    return out


def render(layout: Layout, rules: Rules) -> str:
    """A text picture of a layout, one letter per ship."""
    grid = [["." for _ in range(rules.size)] for _ in range(rules.size)]
    for ship, placement in zip(rules.fleet, layout, strict=True):
        for cell in cells_of(placement):
            row, col = divmod(cell, rules.size)
            grid[row][col] = ship.name[0]
    return "\n".join(" ".join(row) for row in grid)
