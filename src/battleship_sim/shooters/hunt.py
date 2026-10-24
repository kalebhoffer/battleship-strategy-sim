"""Hunt and target: search with some hunt pattern, then work through the neighbors of hits.

Targeting is the classic version. After a hit, the unfired cells above, right, below,
and left of it go on a stack and are fired at (last in, first out) before hunting
resumes. It ignores ship names and sink announcements. Smarter use of feedback is the
probability map's job. Each hunt policy below changes only where the shooter looks when
it has nothing to follow up.
"""

from __future__ import annotations

import random

from battleship_sim.rules import Feedback, Rules, ShotResult


class HuntPolicy:
    def next(self, fired: int) -> int:
        raise NotImplementedError

    def observe(self, cell: int, result: ShotResult) -> None:
        pass


class OrderHunt(HuntPolicy):
    """Fires through a fixed order of cells, skipping ones already fired at."""

    def __init__(self, order: list[int]) -> None:
        self._order = order
        self._next = 0

    def next(self, fired: int) -> int:
        while fired >> self._order[self._next] & 1:
            self._next += 1
        return self._order[self._next]


def random_hunt(rules: Rules, rng: random.Random) -> HuntPolicy:
    order = list(range(rules.cells))
    rng.shuffle(order)
    return OrderHunt(order)


def parity_hunt(rules: Rules, rng: random.Random) -> HuntPolicy:
    """Random cells on one color of a checkerboard (chosen at random), then the rest.

    Every ship is at least two long, so it always covers a cell of each color.
    """
    n = rules.size
    color = rng.randrange(2)
    on = [c for c in range(rules.cells) if (c // n + c % n) % 2 == color]
    off = [c for c in range(rules.cells) if (c // n + c % n) % 2 != color]
    rng.shuffle(on)
    rng.shuffle(off)
    return OrderHunt(on + off)


class AdaptiveParityHunt(HuntPolicy):
    """Hunts on diagonal stripes spaced by the shortest ship still afloat.

    Stripes (row + col) % m == offset meet every run of m cells in a row or column, so
    they are certain to touch every ship of length m or more. When the shortest ship
    sinks, the spacing widens. Ships are known to sink only if the rules announce sinks;
    under hit-only feedback the spacing stays at the shortest ship in the fleet.
    """

    def __init__(self, rules: Rules, rng: random.Random) -> None:
        self._rules = rules
        self._rng = rng
        self._afloat = {i: s.length for i, s in enumerate(rules.fleet)}
        self._offsets = {m: rng.randrange(m) for m in {s.length for s in rules.fleet}}

    @property
    def spacing(self) -> int:
        return min(self._afloat.values())

    def next(self, fired: int) -> int:
        n, m = self._rules.size, self.spacing
        offset = self._offsets[m]
        unfired = [c for c in range(self._rules.cells) if not fired >> c & 1]
        stripe = [c for c in unfired if (c // n + c % n) % m == offset]
        return self._rng.choice(stripe or unfired)

    def observe(self, cell: int, result: ShotResult) -> None:
        if result.sunk is not None and self._rules.feedback is not Feedback.HIT_ONLY:
            self._afloat.pop(result.sunk, None)


def _spiral(n: int) -> list[int]:
    """All cells clockwise from A1, edge first, ending in the middle."""
    out = []
    top, left, bottom, right = 0, 0, n - 1, n - 1
    while top <= bottom and left <= right:
        out += [top * n + c for c in range(left, right + 1)]
        out += [r * n + right for r in range(top + 1, bottom + 1)]
        if top < bottom:
            out += [bottom * n + c for c in range(right - 1, left - 1, -1)]
        if left < right:
            out += [r * n + left for r in range(bottom - 1, top, -1)]
        top, left, bottom, right = top + 1, left + 1, bottom - 1, right - 1
    return out


def pattern_order(pattern: str, n: int) -> list[int]:
    """Fixed sweeps people use. All are deterministic, which a placer could exploit."""
    match pattern:
        case "rows":  # A1, A2, ..., J10
            return list(range(n * n))
        case "diagonal":  # every other diagonal from the top left, then the gaps
            cells = sorted(range(n * n), key=lambda c: (c // n + c % n, c // n))
            return [c for c in cells if (c // n + c % n) % 2 == 0] + [
                c for c in cells if (c // n + c % n) % 2 == 1
            ]
        case "spiral-in":
            return _spiral(n)
        case "spiral-out":
            return _spiral(n)[::-1]
    raise ValueError(f"unknown pattern {pattern!r}")


class HuntTarget:
    def __init__(self, rules: Rules, hunt: HuntPolicy) -> None:
        self._n = rules.size
        self._hunt = hunt
        self._fired = 0
        self._stack: list[int] = []

    @property
    def targeting(self) -> bool:
        while self._stack and self._fired >> self._stack[-1] & 1:
            self._stack.pop()
        return bool(self._stack)

    def fire(self) -> int:
        if self.targeting:
            return self._stack.pop()
        return self._hunt.next(self._fired)

    def observe(self, cell: int, result: ShotResult) -> None:
        self._fired |= 1 << cell
        self._hunt.observe(cell, result)
        if not result.hit:
            return
        n = self._n
        row, col = divmod(cell, n)
        # Pushed in reverse so the stack pops up, right, down, left.
        for r, c in ((row, col - 1), (row + 1, col), (row, col + 1), (row - 1, col)):
            if 0 <= r < n and 0 <= c < n and not self._fired >> (r * n + c) & 1:
                self._stack.append(r * n + c)


def hunt_target(rules: Rules, rng: random.Random) -> HuntTarget:
    return HuntTarget(rules, random_hunt(rules, rng))


def hunt_target_parity(rules: Rules, rng: random.Random) -> HuntTarget:
    return HuntTarget(rules, parity_hunt(rules, rng))


def adaptive_parity(rules: Rules, rng: random.Random) -> HuntTarget:
    return HuntTarget(rules, AdaptiveParityHunt(rules, rng))


def pattern(name: str):
    def factory(rules: Rules, rng: random.Random) -> HuntTarget:
        return HuntTarget(rules, OrderHunt(pattern_order(name, rules.size)))

    factory.__name__ = f"pattern_{name.replace('-', '_')}"
    return factory
