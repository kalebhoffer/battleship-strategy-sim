"""What a shooter has learned so far, under whichever feedback rule is in force.

Under the official rules (named hits) every hit cell is known to belong to a named ship.
Under sink-only feedback, a sunk ship's cells have to be worked out from the hits around
the sinking shot, and may stay ambiguous. Under hit-only feedback nothing is known beyond
hit or miss.
"""

from __future__ import annotations

from battleship_sim.board import fits, placements
from battleship_sim.rules import Feedback, Rules, ShotResult


class Knowledge:
    def __init__(self, rules: Rules) -> None:
        self.rules = rules
        self.fired = 0
        self.misses = 0
        self.hits = 0
        self.sunk: set[int] = set()
        self._sink_cell: dict[int, int] = {}
        # Hit cells as of the shot that sank each ship: a sunk ship lies within these.
        self.hits_at_sink: dict[int, int] = {}
        # known[i]: cells certainly belonging to ship i
        self.known = [0] * len(rules.fleet)

    def record(self, cell: int, result: ShotResult) -> None:
        bit = 1 << cell
        self.fired |= bit
        if not result.hit:
            self.misses |= bit
            return
        self.hits |= bit
        if result.ship is not None and self.rules.feedback is Feedback.NAMED_HITS:
            self.known[result.ship] |= bit
        if result.sunk is not None:
            self.sunk.add(result.sunk)
            self._sink_cell[result.sunk] = cell
            self.hits_at_sink[result.sunk] = self.hits
            if self.rules.feedback is Feedback.SINK_ONLY:
                self._resolve_sunk()

    @property
    def afloat(self) -> list[int]:
        """Fleet indices of ships not known to be sunk."""
        return [i for i in range(len(self.rules.fleet)) if i not in self.sunk]

    def known_cells(self, exclude: int | None = None) -> int:
        out = 0
        for i, mask in enumerate(self.known):
            if i != exclude:
                out |= mask
        return out

    @property
    def unexplained(self) -> int:
        """Hit cells not yet attributed to any ship."""
        return self.hits & ~self.known_cells()

    def sunk_candidates(self, ship: int) -> list[int]:
        """Positions a sunk ship could occupy: through its sinking shot, on squares hit
        by then, clear of cells known to be other ships."""
        cell = self._sink_cell[ship]
        hit_by_then = self.hits_at_sink[ship]
        others = self.known_cells(exclude=ship)
        length = self.rules.fleet[ship].length
        return [
            p
            for p in placements(length, self.rules.size)
            if p >> cell & 1 and p & hit_by_then == p and fits(p, others, self.rules)
        ]

    def _resolve_sunk(self) -> None:
        # A cell is known to be ship i if every remaining position for ship i covers it.
        # Pinning one ship can narrow another, so repeat until nothing changes.
        changed = True
        while changed:
            changed = False
            for ship in self.sunk:
                candidates = self.sunk_candidates(ship)
                if not candidates:
                    continue
                common = candidates[0]
                for p in candidates[1:]:
                    common &= p
                if common != self.known[ship]:
                    self.known[ship] = common
                    changed = True
