"""The hidden ocean and the solve loop: one shooter against one fleet.

The layout lives only in `Ocean`. A shooter is built from the rules and an RNG and
learns about the board solely through the `ShotResult`s passed to `observe`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from battleship_sim.board import Layout, cells_of
from battleship_sim.rules import Feedback, Rules, ShotResult

DEFAULT_MAX_SHOTS = 10_000  # only reachable by shooters that may repeat cells


class Shooter(Protocol):
    def fire(self) -> int: ...

    def observe(self, cell: int, result: ShotResult) -> None: ...


class Ocean:
    """One player's hidden grid. Answers shots the way the rules say to."""

    def __init__(self, layout: Layout, rules: Rules) -> None:
        self._rules = rules
        self._owner: dict[int, int] = {}
        for index, placement in enumerate(layout):
            for cell in cells_of(placement):
                self._owner[cell] = index
        self._left = [ship.length for ship in rules.fleet]
        self._struck: set[int] = set()
        self.afloat = len(rules.fleet)

    def fire(self, cell: int) -> tuple[ShotResult, int | None]:
        """Return what the shooter is told, plus the ship actually sunk (for scoring)."""
        if not 0 <= cell < self._rules.cells:
            raise ValueError(f"shot {cell} is off the grid")
        ship = self._owner.get(cell)
        if ship is None:
            return ShotResult(hit=False), None
        sunk = None
        if cell not in self._struck:
            self._struck.add(cell)
            self._left[ship] -= 1
            if self._left[ship] == 0:
                sunk = ship
                self.afloat -= 1
        match self._rules.feedback:
            case Feedback.NAMED_HITS:
                told = ShotResult(hit=True, ship=ship, sunk=sunk)
            case Feedback.SINK_ONLY:
                told = ShotResult(hit=True, ship=sunk, sunk=sunk)
            case Feedback.HIT_ONLY:
                told = ShotResult(hit=True)
        return told, sunk


@dataclass(frozen=True, slots=True)
class SolveResult:
    shots: int  # shots fired until the last ship sank (or the cap)
    first_hit: int | None  # shot number of the first hit
    sunk_at: tuple[int | None, ...]  # shot number each ship sank, in fleet order
    repeats: int  # shots at cells already fired at
    capped: bool  # hit the shot cap before sinking the fleet


def solve(
    layout: Layout, rules: Rules, shooter: Shooter, max_shots: int = DEFAULT_MAX_SHOTS
) -> SolveResult:
    ocean = Ocean(layout, rules)
    fired: set[int] = set()
    sunk_at: list[int | None] = [None] * len(rules.fleet)
    first_hit = None
    repeats = 0
    shots = 0
    while ocean.afloat and shots < max_shots:
        cell = shooter.fire()
        shots += 1
        if cell in fired:
            repeats += 1
        fired.add(cell)
        told, sunk = ocean.fire(cell)
        if told.hit and first_hit is None:
            first_hit = shots
        if sunk is not None:
            sunk_at[sunk] = shots
        shooter.observe(cell, told)
    return SolveResult(shots, first_hit, tuple(sunk_at), repeats, capped=bool(ocean.afloat))
