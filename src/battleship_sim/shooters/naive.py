"""Shooters that ignore feedback: the baselines every other strategy is measured against."""

from __future__ import annotations

import random

from battleship_sim.rules import Rules, ShotResult


class RandomShooter:
    """Fires at every cell once, in a random order."""

    def __init__(self, rules: Rules, rng: random.Random) -> None:
        self._order = list(range(rules.cells))
        rng.shuffle(self._order)
        self._next = 0

    def fire(self) -> int:
        cell = self._order[self._next]
        self._next += 1
        return cell

    def observe(self, cell: int, result: ShotResult) -> None:
        pass  # deliberately ignores what it learns


class RandomRepeatShooter:
    """Picks any cell uniformly at random every turn, repeats included."""

    def __init__(self, rules: Rules, rng: random.Random) -> None:
        self._cells = rules.cells
        self._rng = rng

    def fire(self) -> int:
        return self._rng.randrange(self._cells)

    def observe(self, cell: int, result: ShotResult) -> None:
        pass  # deliberately ignores what it learns
