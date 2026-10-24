"""Research plan 5.3: counts, placer uniformity, and random fire against exact answers."""

import math
import random
from fractions import Fraction

import pytest

from battleship_sim import exact
from battleship_sim.board import is_legal
from battleship_sim.count import count_layouts, count_layouts_brute, independent_positions
from battleship_sim.placers import get_placer
from battleship_sim.rules import Rules, Ship
from battleship_sim.stats import chi_square_p, gamma_q, ks_statistic
from battleship_sim.validate import (
    SMALL_BOARDS,
    check_count_full,
    check_placers,
    check_random_fire,
    check_random_repeat,
    placer_chi_square,
    small_rules,
)

# Exact 10x10 counts from count_layouts (about 20-30 s each, so stored here and checked
# against an independent Monte Carlo estimate below; `battleship-sim validate` recomputes).
FULL_COUNT_TOUCHING = 30_093_975_536
FULL_COUNT_NO_TOUCHING = 3_851_502_784


@pytest.mark.parametrize(("size", "lengths"), SMALL_BOARDS)
@pytest.mark.parametrize("touching", [True, False])
def test_dynamic_program_matches_brute_force(size, lengths, touching):
    rules = small_rules(size, lengths, touching)
    assert count_layouts(rules) == count_layouts_brute(rules)


@pytest.mark.parametrize("n", [3, 5, 10])
def test_two_single_cells_closed_form(n):
    fleet = (Ship("a", 1), Ship("b", 1))
    assert count_layouts(Rules(size=n, fleet=fleet)) == n * n * (n * n - 1)
    # Ordered pairs that touch: horizontal, vertical, and two diagonal directions.
    touching_pairs = 2 * (2 * n * (n - 1) + 2 * (n - 1) ** 2)
    assert count_layouts(Rules(size=n, fleet=fleet, touching=False)) == (
        n * n * (n * n - 1) - touching_pairs
    )


def test_single_ship_count():
    assert count_layouts(Rules(fleet=(Ship("Carrier", 5),))) == 120


@pytest.mark.parametrize(
    ("touching", "count"), [(True, FULL_COUNT_TOUCHING), (False, FULL_COUNT_NO_TOUCHING)]
)
def test_full_board_count_matches_monte_carlo(touching, count, monkeypatch):
    rules = Rules(touching=touching)
    monkeypatch.setattr("battleship_sim.validate.count_layouts", lambda _: count)
    check = check_count_full(rules, samples=100_000, seed="test")
    assert check.passed, check.detail
    assert independent_positions(rules) == 120 * 140 * 160 * 160 * 180


def test_placers_always_produce_legal_layouts():
    for touching in (True, False):
        rules = Rules(touching=touching)
        for name in ("uniform", "naive"):
            place = get_placer(name)
            rng = random.Random(name)
            for _ in range(300):
                assert is_legal(place(rules, rng), rules)


def test_placer_uniformity_checks():
    for check in check_placers(samples=50_000):
        assert check.passed, f"{check.name}: {check.detail}"


def test_placers_are_reproducible():
    rules = Rules()
    for name in ("uniform", "naive"):
        place = get_placer(name)
        assert place(rules, random.Random("x")) == place(rules, random.Random("x"))


def test_random_fire_formulas_agree_with_the_distribution():
    k, n = 17, 100
    cdf = exact.random_fire_cdf(k, n)
    assert cdf[16] == 0 and cdf[17] > 0 and cdf[100] == 1
    pmf = [cdf[t] - cdf[t - 1] for t in range(1, n + 1)]
    mean = sum(t * p for t, p in zip(range(1, n + 1), pmf, strict=True))
    var = sum((t - mean) ** 2 * p for t, p in zip(range(1, n + 1), pmf, strict=True))
    assert mean == exact.random_fire_mean(k, n) == Fraction(17 * 101, 18)
    assert var == exact.random_fire_variance(k, n)


def test_random_repeat_formula_by_brute_force():
    # Tiny case: 2 ship cells among 3; enumerate the Markov chain directly.
    # E = 3/2 + 3/1 = 4.5
    assert exact.random_repeat_mean(2, 3) == Fraction(9, 2)
    assert exact.random_repeat_mean(17, 100) == 100 * sum(Fraction(1, j) for j in range(1, 18))


def test_simulated_random_fire_matches_exact():
    for check in check_random_fire(Rules(), games=10_000):
        assert check.passed, f"{check.name}: {check.detail}"
    check = check_random_repeat(Rules(), games=3_000)
    assert check.passed, check.detail


def test_random_fire_on_a_variant_board():
    rules = Rules(size=8, fleet=(Ship("a", 4), Ship("b", 3), Ship("c", 2)))
    for check in check_random_fire(rules, games=5_000):
        assert check.passed, f"{check.name}: {check.detail}"


# -- statistics helpers ---------------------------------------------------------


def test_gamma_q_known_values():
    # Chi-square survival function: df=2 is exp(-x/2).
    for x in (0.5, 2.0, 7.0):
        assert gamma_q(1.0, x / 2) == pytest.approx(math.exp(-x / 2))
    # Critical value 3.841 at df=1 gives p = 0.05; 18.307 at df=10 gives p = 0.05.
    assert gamma_q(0.5, 3.841459 / 2) == pytest.approx(0.05, abs=1e-6)
    assert gamma_q(5.0, 18.307038 / 2) == pytest.approx(0.05, abs=1e-6)


def test_chi_square_detects_a_biased_die():
    fair = chi_square_p([100, 98, 103, 99, 101, 99], [100] * 6)[1]
    loaded = chi_square_p([150, 90, 90, 90, 90, 90], [100] * 6)[1]
    assert fair > 0.9
    assert loaded < 0.001


def test_ks_statistic():
    cdf = [0, 0.5, 1.0]
    assert ks_statistic([1, 2], cdf) == 0
    assert ks_statistic([2, 2], cdf) == 0.5


def test_naive_placer_bias_is_real_on_small_board():
    stat, p = placer_chi_square("naive", small_rules(4, (3, 2), True), 50_000, "bias")
    assert p < 0.001
