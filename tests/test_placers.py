"""Phase 3: placement styles, the shooter-placer game, and the worst-case search."""

import random

import pytest

from battleship_sim.board import cells_of, is_legal
from battleship_sim.count import count_layouts
from battleship_sim.matrix import run_matrix, solve_zero_sum, worst_case
from battleship_sim.placers import PLACERS, get_placer
from battleship_sim.placers.styles import _neighbors, opening_odds, spread_score
from battleship_sim.rules import Rules, Ship
from battleship_sim.validate import placer_chi_square, small_rules

STYLES = ["edges", "center", "clustered", "spread", "anti-density"]


def _layouts(name, rules, n=200, seed="t"):
    place = get_placer(name)
    rng = random.Random(seed)
    return [place(rules, rng) for _ in range(n)]


@pytest.mark.parametrize("name", list(PLACERS))
def test_every_placer_is_legal_and_reproducible(name):
    rules = Rules()
    for layout in _layouts(name, rules, 100):
        assert is_legal(layout, rules)
    place = get_placer(name)
    assert place(rules, random.Random(5)) == place(rules, random.Random(5))


@pytest.mark.parametrize("name", ["uniform", "naive", "edges", "spread", "anti-density"])
def test_placers_respect_no_touching(name):
    rules = Rules(touching=False)
    for layout in _layouts(name, rules, 50):
        assert is_legal(layout, rules)


def test_edges_every_ship_on_the_border():
    for layout in _layouts("edges", Rules()):
        for ship in layout:
            assert any(c // 10 in (0, 9) or c % 10 in (0, 9) for c in cells_of(ship))


def test_center_every_ship_in_the_middle():
    for layout in _layouts("center", Rules()):
        for ship in layout:
            assert all(2 <= c // 10 <= 7 and 2 <= c % 10 <= 7 for c in cells_of(ship))


def test_center_without_touching_is_impossible():
    assert count_layouts(Rules(size=6, touching=False)) == 0
    with pytest.raises(ValueError):
        get_placer("center")(Rules(touching=False), random.Random(0))


def test_clustered_forms_one_group():
    for layout in _layouts("clustered", Rules()):
        group = layout[0]
        rest = list(layout[1:])
        while rest:
            joined = [s for s in rest if s & _neighbors(group, 10)]
            assert joined, "a ship is cut off from the group"
            for s in joined:
                group |= s
                rest.remove(s)
    with pytest.raises(ValueError):
        get_placer("clustered")(Rules(touching=False), random.Random(0))


def test_spread_beats_uniform_on_spacing():
    rules = Rules()
    spread = [spread_score(x, 10)[0] for x in _layouts("spread", rules, 100)]
    uniform = [spread_score(x, 10)[0] for x in _layouts("uniform", rules, 100)]
    assert sum(spread) / 100 > sum(uniform) / 100 + 1


def test_anti_density_avoids_the_likely_squares():
    rules = Rules()
    odds = opening_odds(rules)

    def heat(layout):
        return sum(odds[c] for ship in layout for c in cells_of(ship))

    anti = sum(map(heat, _layouts("anti-density", rules, 200))) / 200
    uniform = sum(map(heat, _layouts("uniform", rules, 200))) / 200
    assert anti < uniform * 0.9


def test_edges_is_uniform_among_its_layouts():
    # 4x4, ships 3 and 2: compare with every layout where both ships touch the edge.
    rules = small_rules(4, (3, 2), True)
    def on_edge(layout):
        return all(
            any(c // 4 in (0, 3) or c % 4 in (0, 3) for c in cells_of(ship)) for ship in layout
        )

    _, p = placer_chi_square("edges", rules, 30_000, "edges-uniform", support=on_edge)
    assert p > 0.001


def test_mix():
    rules = Rules()
    always_center = get_placer("mix:center=1,edges=0")
    for layout in [always_center(rules, random.Random(i)) for i in range(20)]:
        assert all(2 <= c // 10 <= 7 for ship in layout for c in cells_of(ship))
    half = get_placer("mix:center=1,edges=1")
    picks = [half(rules, random.Random(i)) for i in range(400)]
    on_edge = sum(any(c // 10 in (0, 9) or c % 10 in (0, 9) for c in cells_of(x[0]))
                  for x in picks)
    assert 120 < on_edge < 280
    for bad in ("mix:center", "mix:center=-1,edges=1", "mix:nope=1"):
        with pytest.raises(ValueError):
            get_placer(bad)


# -- the game -------------------------------------------------------------------------


def test_solver_matching_pennies():
    # Shooter picks a row, placer a column; a fair coin each is the equilibrium, value 1.5.
    eq = solve_zero_sum([[1, 2], [2, 1]], ["a", "b"], ["x", "y"], rounds=20_000)
    assert eq.placer_mix["x"] == pytest.approx(0.5, abs=0.02)
    assert eq.shooter_mix["a"] == pytest.approx(0.5, abs=0.02)
    assert eq.lower <= 1.5 + 1e-9 <= eq.upper + 2e-9
    assert eq.upper - eq.lower < 0.05


def test_solver_dominant_strategies():
    # Shooter "good" is better against everything; placer "hide" is better against everything.
    eq = solve_zero_sum([[40, 50], [60, 70]], ["good", "bad"], ["show", "hide"], rounds=20_000)
    assert eq.shooter_mix["good"] > 0.95
    assert eq.placer_mix["hide"] > 0.95
    assert eq.lower == pytest.approx(50, abs=0.5)


def test_solver_rock_paper_scissors():
    m = [[1, 2, 0], [0, 1, 2], [2, 0, 1]]
    eq = solve_zero_sum(m, ["r", "p", "s"], ["R", "P", "S"], rounds=50_000)
    for v in eq.placer_mix.values():
        assert v == pytest.approx(1 / 3, abs=0.03)
    assert eq.lower - 1e-9 <= 1 <= eq.upper + 1e-9


def test_run_matrix_smoke():
    rules = Rules(size=7, fleet=(Ship("a", 3), Ship("b", 2)))
    result = run_matrix(["random", "hunt-target"], ["uniform", "edges"], 200, "m", rules, workers=1)
    for s in ("random", "hunt-target"):
        for p in ("uniform", "edges"):
            assert 5 <= result["cells"][s][p]["mean"] <= 49
    low, high = result["equilibrium"]["value_bounds"]
    assert low <= high + 1e-6


def test_worst_case_never_gets_easier():
    rules = Rules(size=6, fleet=(Ship("a", 3), Ship("b", 2)))
    result = worst_case("pattern-rows", rules, steps=60, games=1, seed="w")
    assert result["search_mean"] >= result["start_mean"]
    assert is_legal(tuple(result["layout"]), rules)
    # A row-by-row searcher is deterministic, so the re-check agrees exactly.
    assert result["check"]["mean"] == pytest.approx(result["search_mean"])
