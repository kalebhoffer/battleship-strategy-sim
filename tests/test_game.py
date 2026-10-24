import ast
import inspect
import random
from pathlib import Path

import pytest

from battleship_sim import shooters
from battleship_sim.board import parse_cell
from battleship_sim.game import Ocean, solve
from battleship_sim.placers import get_placer
from battleship_sim.rules import Feedback, Rules, Ship, ShotResult
from battleship_sim.shooters import SHOOTERS, make_shooter

RULES = Rules(size=4, fleet=(Ship("Cruiser", 3), Ship("Patrol Boat", 2)))


def _mask(*names):
    out = 0
    for name in names:
        out |= 1 << parse_cell(name, 4)
    return out


LAYOUT = (_mask("A1", "A2", "A3"), _mask("C4", "D4"))


class Scripted:
    """Fires a fixed list of cells and records what it is told."""

    def __init__(self, cells):
        self.cells = [parse_cell(c, 4) for c in cells]
        self.told = []

    def fire(self):
        return self.cells[len(self.told)]

    def observe(self, cell, result):
        self.told.append(result)


def test_solve_counts_shots_hits_and_sinks():
    shooter = Scripted(["B1", "A1", "A2", "C4", "A3", "D4"])
    result = solve(LAYOUT, RULES, shooter)
    assert result.shots == 6
    assert result.first_hit == 2
    assert result.sunk_at == (5, 6)
    assert result.repeats == 0
    assert not result.capped


@pytest.mark.parametrize(
    ("feedback", "expected"),
    [
        (
            Feedback.NAMED_HITS,
            [ShotResult(False), ShotResult(True, 0), ShotResult(True, 0),
             ShotResult(True, 0, 0)],
        ),
        (
            Feedback.SINK_ONLY,
            [ShotResult(False), ShotResult(True), ShotResult(True), ShotResult(True, 0, 0)],
        ),
        (
            Feedback.HIT_ONLY,
            [ShotResult(False), ShotResult(True), ShotResult(True), ShotResult(True)],
        ),
    ],
)
def test_feedback_levels(feedback, expected):
    rules = Rules(size=4, fleet=RULES.fleet, feedback=feedback)
    shooter = Scripted(["B1", "A1", "A2", "A3", "C4", "D4"])
    result = solve(LAYOUT, rules, shooter)
    assert shooter.told[:4] == expected
    # Scoring still knows when every ship sank, even if the shooter was not told.
    assert result.sunk_at == (4, 6)


def test_repeated_hit_does_not_damage_twice():
    ocean = Ocean(LAYOUT, RULES)
    first, sunk = ocean.fire(parse_cell("C4", 4))
    assert first.hit and sunk is None
    again, sunk = ocean.fire(parse_cell("C4", 4))
    assert again.hit and sunk is None
    assert ocean.afloat == 2
    _, sunk = ocean.fire(parse_cell("D4", 4))
    assert sunk == 1


def test_repeats_and_cap():
    shooter = Scripted(["B1"] * 50)
    result = solve(LAYOUT, RULES, shooter, max_shots=10)
    assert result.shots == 10
    assert result.repeats == 9
    assert result.capped
    assert result.first_hit is None
    assert result.sunk_at == (None, None)


def test_off_grid_shot_is_an_error():
    with pytest.raises(ValueError):
        Ocean(LAYOUT, RULES).fire(16)


def test_random_shooter_covers_every_cell_once():
    rules = Rules()
    shooter = make_shooter("random", rules, random.Random(1))
    shots = [shooter.fire() for _ in range(rules.cells)]
    assert sorted(shots) == list(range(rules.cells))


def test_random_shooter_never_needs_more_than_every_cell():
    rules = Rules()
    for seed in range(50):
        shooter = make_shooter("random", rules, random.Random(seed))
        layout = get_placer("uniform")(rules, random.Random(seed))
        result = solve(layout, rules, shooter)
        assert 17 <= result.shots <= 100
        assert result.repeats == 0


def test_unknown_shooter():
    with pytest.raises(ValueError):
        make_shooter("telepathic", Rules(), random.Random(0))


# -- no peeking ---------------------------------------------------------------


def test_shooter_factories_receive_only_rules_and_rng():
    for spec in SHOOTERS.values():
        params = list(inspect.signature(spec.factory).parameters)
        assert params == ["rules", "rng"], spec.name


def test_strategy_modules_cannot_reach_the_hidden_board():
    """Strategy code may use board geometry but must not import the game or placers,
    which are the only places a hidden layout exists."""
    package = Path(shooters.__file__).parent
    for path in package.glob("*.py"):
        if path.name == "__init__.py":
            continue  # the registry imports the Shooter protocol only
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not module.startswith(("battleship_sim.game", "battleship_sim.placers")), (
                    f"{path.name} imports {module}"
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(
                        ("battleship_sim.game", "battleship_sim.placers")
                    ), f"{path.name} imports {alias.name}"
