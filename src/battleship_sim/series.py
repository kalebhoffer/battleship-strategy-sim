"""Series of games against one opponent (Study 8): can a shooter learn its habits?

Each series is a fresh opponent who hides a new fleet every game with the same placement
style. Every shooter in the lineup plays the same fleets, in the same order, and keeps
whatever it learned from one game to the next (shooters with a `new_game()` method; the
others simply start over). Results are reported by game number within the series, so a
learning shooter shows up as a falling line.
"""

from __future__ import annotations

import math
import os
import random
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

from battleship_sim.game import solve
from battleship_sim.placers import get_placer
from battleship_sim.rules import Rules
from battleship_sim.shooters import make_shooter
from battleship_sim.stats import Z95

SeriesResult = dict[str, list[int]]  # shooter -> shots in game 1, 2, ...


@dataclass(frozen=True)
class SeriesConfig:
    shooters: tuple[str, ...]
    placer: str = "uniform"
    series: int = 1_000  # how many opponents
    length: int = 20  # games per opponent
    seed: int | str = 0
    rules: Rules = field(default_factory=Rules)
    workers: int = 0
    chunk_size: int = 50


def play_series(config: SeriesConfig, index: int) -> SeriesResult:
    place = get_placer(config.placer)
    shooters = {
        name: make_shooter(name, config.rules,
                           random.Random(f"{config.seed}:shooter:{name}:{index}"))
        for name in config.shooters
    }
    out: SeriesResult = {name: [] for name in config.shooters}
    for game in range(config.length):
        layout = place(config.rules,
                       random.Random(f"{config.seed}:fleet:{config.placer}:{index}:{game}"))
        for name, shooter in shooters.items():
            if game and hasattr(shooter, "new_game"):
                shooter.new_game()
            elif game:
                shooter = shooters[name] = make_shooter(
                    name, config.rules,
                    random.Random(f"{config.seed}:shooter:{name}:{index}:{game}"))
            out[name].append(solve(layout, config.rules, shooter).shots)
    return out


def _chunk(config: SeriesConfig, start: int, stop: int) -> list[tuple[int, SeriesResult]]:
    return [(i, play_series(config, i)) for i in range(start, stop)]


def run_series(config: SeriesConfig) -> Iterator[tuple[int, SeriesResult]]:
    for name in config.shooters:
        make_shooter(name, config.rules, random.Random(0))
    get_placer(config.placer)
    chunks = [(s, min(s + config.chunk_size, config.series))
              for s in range(0, config.series, config.chunk_size)]
    workers = config.workers or os.cpu_count() or 1
    if workers == 1:
        for a, b in chunks:
            yield from _chunk(config, a, b)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in [pool.submit(_chunk, config, a, b) for a, b in chunks]:
            yield from future.result()


@dataclass
class SeriesTally:
    """Exact running sums of shots by game number, per shooter."""

    length: int
    series: int = 0
    sums: dict[str, list[list[int]]] = field(default_factory=dict)  # name -> [[sum, sumsq]]

    def add(self, result: SeriesResult) -> None:
        self.series += 1
        for name, shots in result.items():
            rows = self.sums.setdefault(name, [[0, 0] for _ in range(self.length)])
            for k, s in enumerate(shots):
                rows[k][0] += s
                rows[k][1] += s * s

    def summary(self) -> dict:
        n = self.series
        out = {}
        for name, rows in self.sums.items():
            by_game = []
            for total, sumsq in rows:
                mean = total / n
                var = (sumsq - total * total / n) / (n - 1) if n > 1 else 0.0
                half = Z95 * math.sqrt(max(var, 0.0) / n) if n > 1 else 0.0
                by_game.append({"mean": mean, "ci95": [mean - half, mean + half]})
            out[name] = by_game
        return {"series": n, "length": self.length, "shooters": out}
