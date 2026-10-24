"""Phase 2 shooters: behavior on hand-built boards, plus every shooter under every rule."""

import random

import pytest

from battleship_sim.board import cell_name, cells_of, parse_cell, placements
from battleship_sim.game import solve
from battleship_sim.placers import get_placer
from battleship_sim.rules import Feedback, Rules, Ship, ShotResult
from battleship_sim.shooters import SHOOTERS, make_shooter
from battleship_sim.shooters.hunt import (
    AdaptiveParityHunt,
    _spiral,
    pattern_order,
)
from battleship_sim.shooters.knowledge import Knowledge

C = lambda name: parse_cell(name, 10)  # noqa: E731


def _mask(*names, size=10):
    out = 0
    for name in names:
        out |= 1 << parse_cell(name, size)
    return out


# -- every shooter, every rule set ------------------------------------------------

RULE_SETS = [
    Rules(),
    Rules(feedback=Feedback.SINK_ONLY),
    Rules(feedback=Feedback.HIT_ONLY),
    Rules(touching=False),
    Rules(touching=False, feedback=Feedback.SINK_ONLY),
    Rules(size=7, fleet=(Ship("a", 4), Ship("b", 3), Ship("c", 2))),
]


# The Bayesian shooters run here with 5 sweeps (same code, faster); test_bayes.py checks
# their beliefs against exact posteriors and runs them at full strength.
FAST = {"bayes": "bayes:5", "info-gain": "info-gain:5"}


@pytest.mark.parametrize("name", [FAST.get(n, n) for n in SHOOTERS if n != "random-repeat"])
@pytest.mark.parametrize("rules", RULE_SETS, ids=lambda r: f"{r.size}-{r.feedback}-{r.touching}")
def test_every_shooter_finishes_without_repeating(name, rules):
    for placer in ("uniform", "naive"):
        for seed in range(15):
            layout = get_placer(placer)(rules, random.Random(f"{placer}{seed}"))
            shooter = make_shooter(name, rules, random.Random(seed))
            result = solve(layout, rules, shooter)
            assert not result.capped
            assert result.repeats == 0
            assert rules.fleet_cells <= result.shots <= rules.cells
            if result.hunt_shots is not None:
                assert result.hunt_shots + result.target_shots == result.shots
                assert result.hunt_hits + result.target_hits == rules.fleet_cells


def test_shooters_are_reproducible():
    rules = Rules()
    layout = get_placer("uniform")(rules, random.Random(1))
    for name in SHOOTERS:
        a = solve(layout, rules, make_shooter(name, rules, random.Random(9)))
        b = solve(layout, rules, make_shooter(name, rules, random.Random(9)))
        assert a == b, name


# -- hunt and target --------------------------------------------------------------


def test_hunt_target_follows_up_a_hit_on_its_neighbors():
    shooter = make_shooter("hunt-target", Rules(), random.Random(0))
    shooter.observe(C("E5"), ShotResult(hit=True, ship=0))
    assert shooter.targeting
    follow_up = [shooter.fire() for _ in range(4)]
    for cell in follow_up:
        shooter.observe(cell, ShotResult(hit=False))
    assert [cell_name(c, 10) for c in follow_up] == ["D5", "E6", "F5", "E4"]
    assert not shooter.targeting


def test_hunt_target_skips_off_grid_and_fired_neighbors():
    shooter = make_shooter("hunt-target", Rules(), random.Random(0))
    shooter.observe(C("A2"), ShotResult(hit=False))
    shooter.observe(C("A1"), ShotResult(hit=True, ship=0))
    assert shooter.fire() == C("B1")


def test_parity_hunts_one_checkerboard_color():
    rules = Rules()
    for seed in range(5):
        shooter = make_shooter("hunt-target-parity", rules, random.Random(seed))
        colors = set()
        for _ in range(50):  # all misses: half the board, one color
            cell = shooter.fire()
            colors.add((cell // 10 + cell % 10) % 2)
            shooter.observe(cell, ShotResult(hit=False))
        assert len(colors) == 1


def test_adaptive_parity_widens_when_small_ships_sink():
    rules = Rules()
    hunt = AdaptiveParityHunt(rules, random.Random(0))
    assert hunt.spacing == 2
    hunt.observe(0, ShotResult(hit=True, ship=4, sunk=4))  # patrol boat
    assert hunt.spacing == 3
    hunt.observe(1, ShotResult(hit=True, ship=2, sunk=2))  # one of the 3s
    assert hunt.spacing == 3
    hunt.observe(2, ShotResult(hit=True, ship=3, sunk=3))  # the other 3
    assert hunt.spacing == 4
    cell = hunt.next(0)
    assert (cell // 10 + cell % 10) % 4 == hunt._offsets[4]


def test_adaptive_parity_cannot_widen_without_sink_news():
    hunt = AdaptiveParityHunt(Rules(feedback=Feedback.HIT_ONLY), random.Random(0))
    hunt.observe(0, ShotResult(hit=True))
    assert hunt.spacing == 2


def test_stripes_meet_every_ship_long_enough():
    for m in (2, 3, 4, 5):
        for offset in range(m):
            stripe = {c for c in range(100) if (c // 10 + c % 10) % m == offset}
            for length in range(m, 6):
                for p in placements(length, 10):
                    assert stripe & set(cells_of(p))


def test_patterns():
    assert pattern_order("rows", 10)[:3] == [0, 1, 2]
    spiral = _spiral(10)
    assert sorted(spiral) == list(range(100))
    assert [cell_name(c, 10) for c in spiral[:12]] == [
        "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "B10", "C10",
    ]
    assert cell_name(pattern_order("spiral-out", 10)[0], 10) in {"E5", "E6", "F5", "F6"}
    diagonal = pattern_order("diagonal", 10)
    assert sorted(diagonal) == list(range(100))
    assert all((c // 10 + c % 10) % 2 == 0 for c in diagonal[:50])
    for n in (1, 2, 3, 5):
        assert sorted(_spiral(n)) == list(range(n * n))


# -- knowledge under each feedback rule -------------------------------------------


def test_knowledge_named_hits():
    k = Knowledge(Rules())
    k.record(C("A1"), ShotResult(True, ship=4))
    k.record(C("A2"), ShotResult(True, ship=4, sunk=4))
    k.record(C("C3"), ShotResult(True, ship=0))
    assert k.known[4] == _mask("A1", "A2")
    assert k.known[0] == _mask("C3")
    assert k.unexplained == 0
    assert k.afloat == [0, 1, 2, 3]


def test_knowledge_sink_only_unique_resolution():
    k = Knowledge(Rules(feedback=Feedback.SINK_ONLY))
    k.record(C("A1"), ShotResult(True))
    k.record(C("B1"), ShotResult(False))
    k.record(C("A2"), ShotResult(True, ship=4, sunk=4))
    # A2 sank the patrol boat; the only all-hit 2-long position through A2 is A1-A2.
    assert k.known[4] == _mask("A1", "A2")
    assert k.unexplained == 0


def test_knowledge_sink_only_ambiguous_then_resolved():
    k = Knowledge(Rules(feedback=Feedback.SINK_ONLY))
    for name in ("C3", "C5"):
        k.record(C(name), ShotResult(True))
    k.record(C("C4"), ShotResult(True, ship=4, sunk=4))
    # Patrol boat is C3-C4 or C4-C5: only C4 is certain.
    assert k.known[4] == _mask("C4")
    assert k.unexplained == _mask("C3", "C5")


def test_knowledge_hit_only_learns_nothing_more():
    k = Knowledge(Rules(feedback=Feedback.HIT_ONLY))
    k.record(C("A1"), ShotResult(True))
    assert k.unexplained == _mask("A1")
    assert k.afloat == [0, 1, 2, 3, 4]


# -- probability map --------------------------------------------------------------


def test_probability_map_opens_near_the_center():
    for seed in range(10):
        shooter = make_shooter("probability-map", Rules(), random.Random(seed))
        cell = shooter.fire()
        row, col = divmod(cell, 10)
        assert 3 <= row <= 6 and 3 <= col <= 6, cell_name(cell, 10)


def test_probability_map_scores_are_symmetric_on_an_empty_board():
    scores = make_shooter("probability-map", Rules(), random.Random(0)).scores()
    for cell in range(100):
        r, c = divmod(cell, 10)
        for mirror in ((r, 9 - c), (9 - r, c), (c, r)):
            assert scores[cell] == pytest.approx(scores[mirror[0] * 10 + mirror[1]])
    # Expected ships per cell sums to the fleet's cell count.
    assert sum(scores) == pytest.approx(17)


def test_probability_map_extends_a_named_line():
    shooter = make_shooter("probability-map", Rules(), random.Random(0))
    shooter.observe(C("E5"), ShotResult(True, ship=1))
    shooter.observe(C("E6"), ShotResult(True, ship=1))
    assert shooter.targeting
    assert cell_name(shooter.fire(), 10) in {"E4", "E7"}


def test_probability_map_follows_up_unnamed_hits():
    shooter = make_shooter("probability-map", Rules(feedback=Feedback.HIT_ONLY),
                           random.Random(0))
    shooter.observe(C("E5"), ShotResult(True))
    assert shooter.targeting
    assert cell_name(shooter.fire(), 10) in {"D5", "F5", "E4", "E6"}


def test_probability_map_avoids_cells_next_to_a_known_ship_when_touching_is_banned():
    rules = Rules(touching=False)
    shooter = make_shooter("probability-map", rules, random.Random(0))
    shooter.observe(C("E5"), ShotResult(True, ship=4))
    shooter.observe(C("E6"), ShotResult(True, ship=4, sunk=4))
    scores = shooter.scores()
    for name in ("D4", "D5", "D6", "D7", "E4", "E7", "F4", "F5", "F6", "F7"):
        assert scores[C(name)] == 0, name
    assert not shooter.targeting


def test_probability_map_never_scores_a_miss():
    shooter = make_shooter("probability-map", Rules(), random.Random(0))
    for name in ("A1", "E5", "J10"):
        shooter.observe(C(name), ShotResult(False))
    scores = shooter.scores()
    assert scores[C("A1")] == scores[C("E5")] == scores[C("J10")] == 0
