"""Probability map: fire where a ship is most likely to be, given everything seen so far.

For each ship still afloat, list every position that fits the evidence: no misses, no
cells known to be another ship, every hit already named for this ship (official rules),
and no touching known ships when touching is banned. Each ship spreads one unit of
probability evenly over its positions. A cell's score is the sum over ships of the chance
that the ship covers it, which is the expected number of ships on that cell if the ships
were placed independently. The shooter fires at the highest-scoring unfired cell and
breaks ties at random.

Treating ships independently is the approximation. The full Bayesian shooter (Phase 4)
models how ships constrain each other.

Under sink-only or hit-only feedback, some hits belong to no known ship. While any exist,
only positions that cover one of those hits are counted, weighted by how many they cover,
so the shooter follows up hits the way the official rules let it do automatically.
"""

from __future__ import annotations

import random

from battleship_sim.board import cells_of, halo, placements
from battleship_sim.rules import Rules, ShotResult
from battleship_sim.shooters.knowledge import Knowledge

_TIE = 1e-9


class ProbabilityMap:
    def __init__(self, rules: Rules, rng: random.Random) -> None:
        self._rules = rules
        self._rng = rng
        self.knowledge = Knowledge(rules)
        self._cells = {
            p: cells_of(p)
            for ship in rules.fleet
            for p in placements(ship.length, rules.size)
        }
        # Optional prior weight per position (OpponentAware sets it); None = all equal.
        self.position_weight: dict[int, float] | None = None

    @property
    def targeting(self) -> bool:
        k = self.knowledge
        return bool(k.unexplained) or any(k.known[i] for i in k.afloat)

    def scores(self) -> list[float]:
        k = self.knowledge
        unexplained = k.unexplained
        scores = self._scores(unexplained) if unexplained else None
        if not scores or not any(scores):
            # Nothing can explain the loose hits (or there are none): plain search.
            scores = self._scores(0)
        return scores

    def _scores(self, unexplained: int) -> list[float]:
        k, rules = self.knowledge, self._rules
        out = [0.0] * rules.cells
        for ship in k.afloat:
            others = k.known_cells(exclude=ship)
            blocked = k.misses | others
            if not rules.touching and others:
                blocked |= halo(others, rules.size)
            # Hits known to be this ship's must all be covered.
            must = k.known[ship]
            weighted: list[tuple[int, float]] = []
            prior = self.position_weight
            for p in placements(rules.fleet[ship].length, rules.size):
                if p & blocked or p & must != must:
                    continue
                if unexplained:
                    weight = (p & unexplained).bit_count()
                    if not weight:
                        continue
                else:
                    weight = 1
                if prior is not None:
                    weight *= prior[p]
                weighted.append((p, weight))
            total = sum(w for _, w in weighted)
            if not total:
                continue
            for p, w in weighted:
                share = w / total
                for cell in self._cells[p]:
                    out[cell] += share
        return out

    def fire(self) -> int:
        fired = self.knowledge.fired
        scores = self.scores()
        best = max(
            (s for c, s in enumerate(scores) if not fired >> c & 1), default=0.0
        )
        choices = [
            c for c, s in enumerate(scores) if not fired >> c & 1 and s >= best - _TIE
        ]
        return self._rng.choice(choices)

    def observe(self, cell: int, result: ShotResult) -> None:
        self.knowledge.record(cell, result)


class OpponentAware(ProbabilityMap):
    """A probability map that learns where one opponent tends to hide ships.

    It plays a series of games against the same opponent (call `new_game()` between
    games). After each game it has hit every ship square, so it knows that layout, and
    it counts how often each square held a ship. Each position then gets a weight: the
    product over its squares of (that square's ship rate) / (the average rate), with the
    rates smoothed toward average by PSEUDO_GAMES imaginary games. With no history every
    weight is 1, so its first game is an ordinary probability map.
    """

    PSEUDO_GAMES = 5

    def __init__(self, rules: Rules, rng: random.Random) -> None:
        super().__init__(rules, rng)
        self.games_seen = 0
        self.ship_counts = [0] * rules.cells

    def new_game(self) -> None:
        k = self.knowledge
        if k.hits:
            self.games_seen += 1
            for c in cells_of(k.hits):
                self.ship_counts[c] += 1
        self.knowledge = Knowledge(self._rules)
        self.position_weight = self._weights() if self.games_seen else None

    def _weights(self) -> dict[int, float]:
        rules = self._rules
        base = rules.fleet_cells / rules.cells
        m = self.PSEUDO_GAMES
        rate = [(n + m * base) / (self.games_seen + m) / base for n in self.ship_counts]
        out = {}
        for p, cells in self._cells.items():
            w = 1.0
            for c in cells:
                w *= rate[c]
            out[p] = w
        return out
