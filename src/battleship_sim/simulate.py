"""Run many games reproducibly, with every shooter facing the same fleets.

Game i draws its fleet from a seed that depends only on (seed, i), so adding or removing
a shooter never changes the fleets the others face. Comparisons are paired by game.
"""

from __future__ import annotations

import csv
import itertools
import os
import random
import re
import time
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from battleship_sim import exact
from battleship_sim.board import Layout
from battleship_sim.game import DEFAULT_MAX_SHOTS, SolveResult, solve
from battleship_sim.placers import get_placer
from battleship_sim.rules import Rules
from battleship_sim.shooters import make_shooter
from battleship_sim.stats import mean_ci, quantile, stdev


@dataclass(frozen=True)
class SimConfig:
    shooters: tuple[str, ...]
    placer: str = "uniform"
    games: int = 10_000
    seed: int | str = 0
    rules: Rules = field(default_factory=Rules)
    max_shots: int = DEFAULT_MAX_SHOTS
    workers: int = 0  # 0 = one per CPU
    chunk_size: int = 1_000


def fleet_for(config: SimConfig, game_index: int) -> Layout:
    rng = random.Random(f"{config.seed}:fleet:{config.placer}:{game_index}")
    return get_placer(config.placer)(config.rules, rng)


def play_game(
    config: SimConfig, game_index: int, seconds: dict[str, float] | None = None
) -> list[tuple[str, SolveResult]]:
    """Every shooter against game `game_index`'s fleet. Adds wall-clock time to `seconds`."""
    layout = fleet_for(config, game_index)
    out = []
    for name in config.shooters:
        began = time.perf_counter()
        rng = random.Random(f"{config.seed}:shooter:{name}:{game_index}")
        shooter = make_shooter(name, config.rules, rng)
        out.append((name, solve(layout, config.rules, shooter, config.max_shots)))
        if seconds is not None:
            seconds[name] += time.perf_counter() - began
    return out


Chunk = tuple[list[tuple[int, list[tuple[str, SolveResult]]]], dict[str, float]]


def _play_chunk(config: SimConfig, start: int, stop: int) -> Chunk:
    seconds: dict[str, float] = defaultdict(float)
    games = [(i, play_game(config, i, seconds)) for i in range(start, stop)]
    return games, dict(seconds)


def run(config: SimConfig, timing: dict[str, float] | None = None) -> Iterator[
    tuple[int, list[tuple[str, SolveResult]]]
]:
    """Yield (game_index, [(shooter, result), ...]) for every game, in game order.

    If `timing` is given, it is filled with total wall-clock seconds per shooter.
    """
    for name in config.shooters:
        make_shooter(name, config.rules, random.Random(0))  # fail fast on typos
    get_placer(config.placer)
    chunks = [
        (start, min(start + config.chunk_size, config.games))
        for start in range(0, config.games, config.chunk_size)
    ]
    workers = config.workers or os.cpu_count() or 1

    def collect(chunk: Chunk) -> list[tuple[int, list[tuple[str, SolveResult]]]]:
        games, seconds = chunk
        if timing is not None:
            for name, s in seconds.items():
                timing[name] = timing.get(name, 0.0) + s
        return games

    if workers == 1:
        for start, stop in chunks:
            yield from collect(_play_chunk(config, start, stop))
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_play_chunk, config, start, stop) for start, stop in chunks]
        for future in futures:
            yield from collect(future.result())


def ship_columns(rules: Rules) -> list[str]:
    return [f"sunk_{re.sub(r'[^a-z0-9]+', '_', s.name.lower()).strip('_')}" for s in rules.fleet]


def csv_fields(rules: Rules) -> list[str]:
    return [
        "game", "shooter", "shots", "first_hit", "repeats", "capped",
        *MODE_COLUMNS, *ship_columns(rules),
    ]


MODE_COLUMNS = ("hunt_shots", "hunt_hits", "target_shots", "target_hits")


def result_row(game: int, shooter: str, r: SolveResult, rules: Rules) -> dict[str, object]:
    row: dict[str, object] = {
        "game": game,
        "shooter": shooter,
        "shots": r.shots,
        "first_hit": "" if r.first_hit is None else r.first_hit,
        "repeats": r.repeats,
        "capped": int(r.capped),
    }
    for column in MODE_COLUMNS:
        value = getattr(r, column)
        row[column] = "" if value is None else value
    for column, at in zip(ship_columns(rules), r.sunk_at, strict=True):
        row[column] = "" if at is None else at
    return row


def run_to_csv(
    config: SimConfig, path: Path, progress=None, timing: dict[str, float] | None = None
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields(config.rules))
        writer.writeheader()
        for done, (i, results) in enumerate(run(config, timing), start=1):
            for name, r in results:
                writer.writerow(result_row(i, name, r, config.rules))
            if progress:
                progress(done, config.games)
    return path


# -- summary ---------------------------------------------------------------


def summarize(csv_path: Path, rules: Rules) -> dict[str, object]:
    """Shot-count distribution per shooter, exact values where known, and paired
    differences between every pair of shooters on the same fleets.

    Stores two compact arrays per shooter and running sums for everything else, so a
    million-game run summarizes in modest memory.
    """
    games: dict[str, array] = defaultdict(lambda: array("l"))
    shots: dict[str, array] = defaultdict(lambda: array("l"))
    sums: dict[str, Counter[str]] = defaultdict(Counter)  # column -> total
    counts: dict[str, Counter[str]] = defaultdict(Counter)  # column -> rows with a value
    columns = ship_columns(rules)
    summed = ("first_hit", "capped", *MODE_COLUMNS, *columns)
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            name = row["shooter"]
            games[name].append(int(row["game"]))
            shots[name].append(int(row["shots"]))
            for column in summed:
                if row[column]:
                    sums[name][column] += int(row[column])
                    counts[name][column] += 1

    def average(name: str, column: str) -> float | None:
        n = counts[name][column]
        return sums[name][column] / n if n else None

    shooters = {}
    for name, values in shots.items():
        ordered = sorted(values)
        mean, low, high = mean_ci(values)
        entry: dict[str, object] = {
            "games": len(values),
            "mean": mean,
            "ci95": [low, high],
            "sd": stdev(values, mean),
            "median": quantile(ordered, 0.5),
            "p5": quantile(ordered, 0.05),
            "p95": quantile(ordered, 0.95),
            "min": ordered[0],
            "max": ordered[-1],
            "capped": sums[name]["capped"],
            "mean_first_hit": average(name, "first_hit"),
            "mean_sunk_at": {c: average(name, c) for c in columns},
        }
        if counts[name]["hunt_shots"]:
            m = sums[name]
            entry["hunt_hit_rate"] = m["hunt_hits"] / m["hunt_shots"] if m["hunt_shots"] else None
            entry["target_hit_rate"] = (
                m["target_hits"] / m["target_shots"] if m["target_shots"] else None
            )
            entry["share_of_shots_targeting"] = m["target_shots"] / sum(values)
        known = exact_mean(name, rules)
        if known is not None:
            entry["exact_mean"] = known
        shooters[name] = entry

    paired = []
    for a, b in itertools.combinations(shots, 2):
        if games[a] == games[b]:  # the normal case: rows in the same game order
            diffs = array("l", (x - y for x, y in zip(shots[a], shots[b], strict=True)))
        else:
            other = dict(zip(games[b], shots[b], strict=True))
            diffs = array("l", (x - other[g] for g, x in zip(games[a], shots[a], strict=True)
                                if g in other))
        mean, low, high = mean_ci(diffs)
        paired.append({"a": a, "b": b, "games": len(diffs), "mean_a_minus_b": mean,
                       "ci95": [low, high]})
    return {"shooters": shooters, "paired": paired}


def exact_mean(shooter: str, rules: Rules) -> float | None:
    """The known expected shot count for shooters that ignore feedback."""
    if shooter == "random":
        return float(exact.random_fire_mean(rules.fleet_cells, rules.cells))
    if shooter == "random-repeat":
        return float(exact.random_repeat_mean(rules.fleet_cells, rules.cells))
    return None
