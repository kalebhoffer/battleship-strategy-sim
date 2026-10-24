"""The shooter-placer game (Study 5) and the worst-case search for one shooter.

The payoff for a shooter against a placer is the average number of shots it needs. The
placer wants it high and the shooter wants it low, so this is a zero-sum game in which
mixed strategies are allowed: a placer can pick a style at random each game. Because the
payoff is an average, the value of any mix is the same mix of averages.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from battleship_sim.board import Layout, fits, placements
from battleship_sim.game import solve
from battleship_sim.placers import get_placer
from battleship_sim.rules import Rules
from battleship_sim.shooters import make_shooter
from battleship_sim.simulate import SimConfig, run
from battleship_sim.stats import mean_ci


@dataclass(frozen=True)
class Equilibrium:
    placer_mix: dict[str, float]  # how often the placer should use each style
    shooter_mix: dict[str, float]  # how often the shooter should use each strategy
    lower: float  # the placer can guarantee at least this many shots on average
    upper: float  # the shooter can hold it to at most this many


def solve_zero_sum(
    payoff: list[list[float]], shooters: list[str], placers: list[str], rounds: int = 200_000
) -> Equilibrium:
    """Approximate equilibrium of the game payoff[shooter][placer] (shots; placer
    maximizes) by multiplicative weights. `lower` and `upper` bracket the true value: the
    averaged strategies guarantee them, so the gap shows how close the answer is."""
    rows, cols = len(shooters), len(placers)
    span = max(max(r) for r in payoff) - min(min(r) for r in payoff) or 1.0
    eta = (8 * max(1.0, (rows * cols) ** 0.5) / rounds) ** 0.5
    w_shooter = [1.0] * rows
    w_placer = [1.0] * cols
    avg_shooter = [0.0] * rows
    avg_placer = [0.0] * cols
    for _ in range(rounds):
        ts, tp = sum(w_shooter), sum(w_placer)
        ps = [w / ts for w in w_shooter]
        pp = [w / tp for w in w_placer]
        for i in range(rows):
            avg_shooter[i] += ps[i]
        for j in range(cols):
            avg_placer[j] += pp[j]
        # Expected shots of each pure strategy against the opponent's current mix.
        vs_placer = [sum(payoff[i][j] * pp[j] for j in range(cols)) for i in range(rows)]
        vs_shooter = [sum(payoff[i][j] * ps[i] for i in range(rows)) for j in range(cols)]
        w_shooter = [w * math.exp(-eta * v / span) for w, v in
                     zip(w_shooter, vs_placer, strict=True)]
        w_placer = [w * math.exp(eta * v / span) for w, v in
                    zip(w_placer, vs_shooter, strict=True)]
        top_s, top_p = max(w_shooter), max(w_placer)
        w_shooter = [w / top_s for w in w_shooter]
        w_placer = [w / top_p for w in w_placer]
    shooter_mix = [x / rounds for x in avg_shooter]
    placer_mix = [x / rounds for x in avg_placer]
    lower = min(sum(payoff[i][j] * placer_mix[j] for j in range(cols)) for i in range(rows))
    upper = max(sum(payoff[i][j] * shooter_mix[i] for i in range(rows)) for j in range(cols))
    return Equilibrium(
        placer_mix=dict(zip(placers, placer_mix, strict=True)),
        shooter_mix=dict(zip(shooters, shooter_mix, strict=True)),
        lower=lower,
        upper=upper,
    )


def run_matrix(
    shooters: list[str], placers: list[str], games: int, seed: str, rules: Rules,
    workers: int = 0,
) -> dict:
    """Every shooter against every placer. Within a placer, shooters face the same fleets."""
    cells: dict[str, dict[str, dict]] = {s: {} for s in shooters}
    for placer in placers:
        config = SimConfig(tuple(shooters), placer, games, seed, rules, workers=workers)
        shots: dict[str, list[int]] = {s: [] for s in shooters}
        for _, results in run(config):
            for name, r in results:
                shots[name].append(r.shots)
        for name in shooters:
            mean, low, high = mean_ci(shots[name])
            cells[name][placer] = {"mean": mean, "ci95": [low, high]}
    payoff = [[cells[s][p]["mean"] for p in placers] for s in shooters]
    eq = solve_zero_sum(payoff, shooters, placers)
    return {
        "shooters": shooters,
        "placers": placers,
        "games_per_cell": games,
        "seed": seed,
        "rules": rules.describe(),
        "cells": cells,
        "equilibrium": {
            "placer_mix": eq.placer_mix,
            "shooter_mix": eq.shooter_mix,
            "value_bounds": [eq.lower, eq.upper],
        },
    }


# -- worst case ----------------------------------------------------------------


def _average_shots(layout: Layout, shooter: str, rules: Rules, seeds: list[str]) -> float:
    total = 0
    for s in seeds:
        total += solve(layout, rules, make_shooter(shooter, rules, random.Random(s))).shots
    return total / len(seeds)


def worst_case(
    shooter: str, rules: Rules, steps: int = 300, games: int = 50, seed: str = "worst",
    log=None,
) -> dict:
    """Hill-climb toward a layout that makes `shooter` take as many shots as possible.

    Each step moves one ship to a random legal spot and keeps the move if the shooter's
    average over the same `games` shooter seeds doesn't drop. Finds a bad layout for that
    shooter, not necessarily the worst one. The result is re-checked on fresh seeds.
    """
    rng = random.Random(f"{seed}:search")
    seeds = [f"{seed}:shot:{i}" for i in range(games)]
    layout = get_placer("uniform")(rules, rng)
    start = best = _average_shots(layout, shooter, rules, seeds)
    for step in range(steps):
        i = rng.randrange(len(layout))
        others = 0
        for k, p in enumerate(layout):
            if k != i:
                others |= p
        room = [p for p in placements(rules.fleet[i].length, rules.size)
                if fits(p, others, rules) and p != layout[i]]
        if not room:
            continue
        candidate = tuple(rng.choice(room) if k == i else p for k, p in enumerate(layout))
        score = _average_shots(candidate, shooter, rules, seeds)
        if score >= best:
            layout, best = candidate, score
        if log and (step + 1) % 50 == 0:
            log(f"  step {step + 1}/{steps}: {best:.2f} shots")
    fresh = [f"{seed}:check:{i}" for i in range(max(games, 200))]
    check = [solve(layout, rules, make_shooter(shooter, rules, random.Random(s))).shots
             for s in fresh]
    mean, low, high = mean_ci(check)
    return {
        "shooter": shooter,
        "layout": list(layout),
        "start_mean": start,
        "search_mean": best,
        "check": {"games": len(check), "mean": mean, "ci95": [low, high], "max": max(check)},
        "steps": steps,
        "games": games,
        "seed": seed,
        "rules": rules.describe(),
    }

