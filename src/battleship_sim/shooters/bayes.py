"""Full Bayesian shooters: reason about whole fleets, not one ship at a time.

Before any shot, every legal fleet layout is equally likely (that is how the uniform
placer hides ships). After each shot, the belief is that same distribution restricted to
the layouts that would have produced exactly what the shooter has been told. Unlike the
probability map, this accounts for how ships crowd each other and for every rule the
feedback implies:

- no ship covers a miss, and every hit is covered by some ship;
- official rules: each ship covers every hit named for it and none named for another;
- a sunk ship lies entirely on squares hit by the shot that sank it, through that shot
  (sink-only rules) or exactly on its named hits (official rules);
- a ship not yet announced sunk is not completely hit (when sinks are announced).

The belief is sampled with a Gibbs sampler: repeatedly pick a ship and redraw its
position uniformly among the positions that keep the whole layout consistent. Each step
also records the full distribution it drew from (Rao-Blackwellization), so the estimated
chance of each ship being at each position is much less noisy than counting samples.
The sampler is checked against exact posteriors on small boards (tests/test_bayes.py).

Two decision rules use the same belief:
- `bayes`: fire where a ship is most likely to be.
- `info-gain`: fire where the answer is most uncertain, i.e. the shot whose outcome has
  the highest entropy, which is the expected information it gains.
"""

from __future__ import annotations

import math
import random

from battleship_sim.board import cells_of, fits, placements
from battleship_sim.rules import Feedback, Rules, ShotResult
from battleship_sim.shooters.knowledge import Knowledge

DEFAULT_SWEEPS = 40
BURN_IN = 5
MAX_SEARCH_NODES = 2_000_000
_TIE = 1e-9


class Belief:
    """Gibbs sampler over fleet layouts consistent with a shooter's knowledge."""

    def __init__(self, rules: Rules, rng: random.Random, sweeps: int = DEFAULT_SWEEPS) -> None:
        self.rules = rules
        self.rng = rng
        self.sweeps = sweeps
        self.knowledge = Knowledge(rules)
        self.state: list[int] | None = None
        self._cells = {
            p: cells_of(p) for s in rules.fleet for p in placements(s.length, rules.size)
        }

    # -- consistency ----------------------------------------------------------------

    def ship_options(self) -> list[list[int]]:
        """Each ship's positions that fit its own evidence (ignoring the other ships)."""
        k, rules = self.knowledge, self.rules
        named = rules.feedback is Feedback.NAMED_HITS
        sinks = rules.feedback is not Feedback.HIT_ONLY
        out = []
        for i, ship in enumerate(rules.fleet):
            options = []
            for p in placements(ship.length, rules.size):
                if p & k.misses:
                    continue
                if named:
                    own = k.known[i]
                    if p & own != own or p & (k.hits & ~own):
                        continue
                if i in k.sunk:
                    if p & k.hits_at_sink[i] != p or not p >> k._sink_cell[i] & 1:
                        continue
                elif sinks and p & k.hits == p:
                    continue  # would have been announced sunk
                options.append(p)
            out.append(options)
        return out

    def consistent(self, layout: list[int], options: list[list[int]]) -> bool:
        taken = 0
        for i, p in enumerate(layout):
            if p not in set(options[i]) or not fits(p, taken, self.rules):
                return False
            taken |= p
        return self.knowledge.hits & ~taken == 0

    def _search(self, options: list[list[int]]) -> list[int]:
        """A random consistent layout, by backtracking.

        First covers the hits: take the uncovered hit with the fewest (ship, position)
        pairs that could cover it and branch on those. Then places the remaining ships,
        most constrained first. Only a starting point: the Gibbs sweeps that follow
        remove its bias.
        """
        hits, rules, rng = self.knowledge.hits, self.rules, self.rng
        layout = [0] * len(options)
        nodes = 0

        def place(remaining: list[int], taken: int) -> bool:
            nonlocal nodes
            nodes += 1
            if nodes > MAX_SEARCH_NODES:
                raise RuntimeError("no consistent layout found within the search budget")
            uncovered = hits & ~taken
            if uncovered:
                best = None
                for c in cells_of(uncovered):
                    pairs = [(i, p) for i in remaining for p in options[i]
                             if p >> c & 1 and fits(p, taken, rules)]
                    if best is None or len(pairs) < len(best):
                        best = pairs
                    if not pairs:
                        return False
                rng.shuffle(best)
                for i, p in best:
                    layout[i] = p
                    if place([j for j in remaining if j != i], taken | p):
                        return True
                return False
            if not remaining:
                return True
            i = min(remaining, key=lambda j: len(options[j]))
            choices = [p for p in options[i] if fits(p, taken, rules)]
            rng.shuffle(choices)
            rest = [j for j in remaining if j != i]
            for p in choices:
                layout[i] = p
                if place(rest, taken | p):
                    return True
            return False

        if not place(list(range(len(options))), 0):
            raise RuntimeError("no fleet layout matches what the shooter has seen")
        return layout

    # -- sampling -------------------------------------------------------------------

    def marginals(self) -> list[dict[int, float]]:
        """For each ship, the estimated probability of each of its positions."""
        options = self.ship_options()
        if self.state is None or not self.consistent(self.state, options):
            self.state = self._search(options)
        state = self.state
        hits = self.knowledge.hits
        weights: list[dict[int, float]] = [{} for _ in options]
        rng, rules = self.rng, self.rules
        n = len(options)
        # Hits not tied to a named ship: one-ship moves alone may not mix (see _pair_move).
        loose = bool(self.knowledge.unexplained)
        for sweep in range(BURN_IN + self.sweeps):
            record = sweep >= BURN_IN
            for i in range(n):
                others = 0
                for j in range(n):
                    if j != i:
                        others |= state[j]
                need = hits & ~others  # hits only ship i can still cover
                valid = [
                    p for p in options[i] if p & need == need and fits(p, others, rules)
                ]
                if record:
                    share = 1.0 / len(valid)
                    w = weights[i]
                    for p in valid:
                        w[p] = w.get(p, 0.0) + share
                state[i] = rng.choice(valid)
            if loose and n > 1:
                for _ in range(n):
                    self._pair_move(state, options)
        return [{p: v / self.sweeps for p, v in w.items()} for w in weights]

    def _pair_move(self, state: list[int], options: list[list[int]]) -> None:
        """Redraw two ships together, uniformly among all valid pairs of positions.

        One-ship moves can freeze when hits aren't named: a ship covering a hit can't
        leave it and no other ship can take it over. Moving two at once lets ships trade
        hits. The pair is chosen uniformly at random: choosing it based on the current
        layout (say, a ship sitting on a hit) would bias the sampler.
        """
        rng, rules, hits = self.rng, self.rules, self.knowledge.hits
        n = len(state)
        i, j = rng.sample(range(n), 2)
        others = 0
        for k in range(n):
            if k not in (i, j):
                others |= state[k]
        need = hits & ~others
        pairs = []
        for pi in options[i]:
            if not fits(pi, others, rules):
                continue
            rest = need & ~pi
            taken = others | pi
            for pj in options[j]:
                if pj & rest == rest and fits(pj, taken, rules):
                    pairs.append((pi, pj))
        state[i], state[j] = rng.choice(pairs)

    def occupancy(self, marginals: list[dict[int, float]]) -> list[float]:
        out = [0.0] * self.rules.cells
        for w in marginals:
            for p, prob in w.items():
                for c in self._cells[p]:
                    out[c] += prob
        return out


class _BayesShooter:
    def __init__(self, rules: Rules, rng: random.Random, sweeps: int = DEFAULT_SWEEPS) -> None:
        self._rng = rng
        self.belief = Belief(rules, rng, sweeps)

    @property
    def knowledge(self) -> Knowledge:
        return self.belief.knowledge

    @property
    def targeting(self) -> bool:
        k = self.knowledge
        hit_ships = any(k.known[i] for i in k.afloat)
        return bool(k.unexplained) or hit_ships

    def observe(self, cell: int, result: ShotResult) -> None:
        self.knowledge.record(cell, result)

    def _pick(self, scores: list[float]) -> int:
        fired = self.knowledge.fired
        best = max((s for c, s in enumerate(scores) if not fired >> c & 1), default=0.0)
        choices = [c for c, s in enumerate(scores) if not fired >> c & 1 and s >= best - _TIE]
        return self._rng.choice(choices)


class BayesShooter(_BayesShooter):
    """Fires at the square most likely to hold a ship."""

    def fire(self) -> int:
        b = self.belief
        return self._pick(b.occupancy(b.marginals()))


class InfoGainShooter(_BayesShooter):
    """Fires where the answer is most uncertain (highest outcome entropy). Ties go to the
    square more likely to be a hit."""

    def fire(self) -> int:
        b = self.belief
        marginals = b.marginals()
        scores = outcome_entropy(marginals, self.knowledge, b.rules)
        occupancy = b.occupancy(marginals)
        # Entropy first; a tiny nudge toward likely hits breaks ties.
        return self._pick([h + 1e-6 * occupancy[c] for c, h in enumerate(scores)])


def outcome_entropy(marginals: list[dict[int, float]], k: Knowledge, rules: Rules) -> list[float]:
    """Entropy (bits) of what the defender would say for a shot at each square."""
    n = rules.cells
    hit = [[0.0] * n for _ in marginals]  # ship i there, not sunk by this shot
    sink = [[0.0] * n for _ in marginals]  # ship i there, sunk by this shot
    for i, w in enumerate(marginals):
        for p, prob in w.items():
            left = p & ~k.hits
            last = left.bit_length() - 1 if left and left & (left - 1) == 0 else -1
            for c in cells_of(left):
                (sink if c == last else hit)[i][c] += prob
    out = [0.0] * n
    for c in range(n):
        if k.fired >> c & 1:
            continue
        if rules.feedback is Feedback.NAMED_HITS:
            outcomes = [hit[i][c] for i in range(len(marginals))]
            outcomes += [sink[i][c] for i in range(len(marginals))]
        elif rules.feedback is Feedback.SINK_ONLY:
            outcomes = [sum(hit[i][c] for i in range(len(marginals)))]
            outcomes += [sink[i][c] for i in range(len(marginals))]
        else:
            outcomes = [sum(hit[i][c] + sink[i][c] for i in range(len(marginals)))]
        miss = max(0.0, 1.0 - sum(outcomes))
        out[c] = -sum(q * math.log2(q) for q in (*outcomes, miss) if q > 1e-12)
    return out


def bayes(sweeps: int = DEFAULT_SWEEPS):
    def factory(rules: Rules, rng: random.Random) -> BayesShooter:
        return BayesShooter(rules, rng, sweeps)

    return factory


def info_gain(sweeps: int = DEFAULT_SWEEPS):
    def factory(rules: Rules, rng: random.Random) -> InfoGainShooter:
        return InfoGainShooter(rules, rng, sweeps)

    return factory
