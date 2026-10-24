"""The checks in research-plan section 5.3. Each must pass before any results count."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from battleship_sim import exact
from battleship_sim.board import fits, placements
from battleship_sim.count import (
    count_layouts,
    count_layouts_brute,
    enumerate_layouts,
    independent_positions,
)
from battleship_sim.game import solve
from battleship_sim.placers import get_placer
from battleship_sim.rules import Rules, Ship
from battleship_sim.shooters import make_shooter
from battleship_sim.stats import chi_square_p, ks_statistic, mean_ci, wilson_interval

# Small boards where every layout can be listed: (size, ship lengths).
SMALL_BOARDS = ((4, (3, 2)), (5, (3, 2, 2)), (5, (3, 3, 2, 1)), (6, (4, 3, 2)))


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def small_rules(size: int, lengths: tuple[int, ...], touching: bool) -> Rules:
    fleet = tuple(Ship(f"Ship {i + 1}", n) for i, n in enumerate(lengths))
    return Rules(size=size, fleet=fleet, touching=touching)


def check_counts_small() -> Iterator[Check]:
    """The dynamic program agrees with listing every layout one by one."""
    for size, lengths in SMALL_BOARDS:
        for touching in (True, False):
            rules = small_rules(size, lengths, touching)
            fast, brute = count_layouts(rules), count_layouts_brute(rules)
            yield Check(
                f"layout count {size}x{size} {lengths} touching={touching}",
                fast == brute,
                f"dynamic program {fast:,}, brute force {brute:,}",
            )


def check_count_full(rules: Rules, samples: int, seed: int | str = "count") -> Check:
    """The exact count agrees with a Monte Carlo estimate: the share of independently
    drawn ship positions that happen to be legal, times all possible position tuples."""
    total = count_layouts(rules)
    positions = independent_positions(rules)
    # Draws positions directly rather than through a placer, so it shares no code with them.
    options = [placements(ship.length, rules.size) for ship in rules.fleet]
    rng = random.Random(seed)
    legal = 0
    for _ in range(samples):
        taken = 0
        for choices in options:
            p = rng.choice(choices)
            if not fits(p, taken, rules):
                break
            taken |= p
        else:
            legal += 1
    low, high = wilson_interval(legal, samples, z=3.29)  # 99.9%
    share = total / positions
    return Check(
        f"layout count {rules.size}x{rules.size} touching={rules.touching}",
        low <= share <= high,
        f"exact {total:,} = {share:.4%} of {positions:,} position tuples; "
        f"sampled {legal / samples:.4%} (99.9% CI {low:.4%} - {high:.4%})",
    )


def _solve_many(shooter: str, rules: Rules, games: int, seed: str) -> list[int]:
    place = get_placer("uniform")
    out = []
    for i in range(games):
        layout = place(rules, random.Random(f"{seed}:fleet:{i}"))
        s = make_shooter(shooter, rules, random.Random(f"{seed}:shot:{i}"))
        out.append(solve(layout, rules, s).shots)
    return out


def check_random_fire(rules: Rules, games: int) -> list[Check]:
    k, n = rules.fleet_cells, rules.cells
    shots = sorted(_solve_many("random", rules, games, "validate-random"))
    want = float(exact.random_fire_mean(k, n))
    se = math.sqrt(float(exact.random_fire_variance(k, n)) / games)
    got = mean_ci(shots)[0]
    ks = ks_statistic(shots, exact.random_fire_cdf(k, n))
    cutoff = 1.95 / math.sqrt(games)  # 0.1% level
    return [
        Check(
            "random fire: mean shots",
            abs(got - want) < 4 * se,
            f"simulated {got:.3f}, exact {want:.3f} (standard error {se:.3f})",
        ),
        Check(
            "random fire: whole distribution",
            ks < cutoff,
            f"largest CDF gap {ks:.4f}, limit {cutoff:.4f}",
        ),
    ]


def check_random_repeat(rules: Rules, games: int) -> Check:
    k, n = rules.fleet_cells, rules.cells
    shots = _solve_many("random-repeat", rules, games, "validate-repeat")
    want = float(exact.random_repeat_mean(k, n))
    se = math.sqrt(float(exact.random_repeat_variance(k, n)) / games)
    got = mean_ci(shots)[0]
    return Check(
        "random fire with repeats: mean shots",
        abs(got - want) < 4 * se,
        f"simulated {got:.2f}, exact {want:.2f} (standard error {se:.2f})",
    )


def placer_chi_square(placer: str, rules: Rules, samples: int, seed: str) -> tuple[float, float]:
    """Chi-square test of a placer against 'every legal layout equally likely'."""
    layouts = enumerate_layouts(rules)
    index = {layout: i for i, layout in enumerate(layouts)}
    place = get_placer(placer)
    rng = random.Random(seed)
    counts = Counter(index[place(rules, rng)] for _ in range(samples))
    observed = [counts.get(i, 0) for i in range(len(layouts))]
    return chi_square_p(observed, [samples / len(layouts)] * len(layouts))


def check_placers(samples: int) -> list[Check]:
    out = []
    for touching in (True, False):
        rules = small_rules(4, (3, 2), touching)
        stat, p = placer_chi_square("uniform", rules, samples, "validate-uniform")
        out.append(
            Check(
                f"uniform placer is uniform (4x4, touching={touching})",
                p > 0.001,
                f"chi-square {stat:.1f}, p = {p:.3f}",
            )
        )
        stat, p = placer_chi_square("naive", rules, samples, "validate-naive")
        out.append(
            Check(
                f"naive placer is detectably not uniform (4x4, touching={touching})",
                p < 0.001,
                f"chi-square {stat:.1f}, p = {p:.2g}",
            )
        )
    return out


def run_all(
    games: int = 20_000, samples: int = 200_000, full_counts: bool = True
) -> Iterator[Check]:
    """Every check, in order. `full_counts` adds the slow 10x10 exact counts (~1 min)."""
    steps: list[Callable[[], Iterator[Check] | list[Check] | Check]] = [
        check_counts_small,
        lambda: check_placers(samples // 4),
        lambda: check_random_fire(Rules(), games),
        lambda: check_random_repeat(Rules(), games // 4),
    ]
    if full_counts:
        steps += [
            lambda: check_count_full(Rules(), samples),
            lambda: check_count_full(Rules(touching=False), samples),
        ]
    for step in steps:
        result = step()
        if isinstance(result, Check):
            yield result
        else:
            yield from result
