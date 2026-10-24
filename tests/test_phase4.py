"""Phase 4: two-player races, Salvo, series against one opponent, and learning."""

import random

import pytest

from battleship_sim import exact
from battleship_sim.board import parse_cell
from battleship_sim.cli import main
from battleship_sim.race import RaceConfig, RaceTally, race, run_races
from battleship_sim.rules import Rules, Ship
from battleship_sim.series import SeriesConfig, SeriesTally, play_series, run_series
from battleship_sim.shooters import make_shooter
from battleship_sim.shooters.probability import OpponentAware

TINY = Rules(size=4, fleet=(Ship("A", 2),))


class Scripted:
    def __init__(self, cells, size=4):
        self.cells = [parse_cell(c, size) for c in cells]
        self.n = 0

    def fire(self):
        self.n += 1
        return self.cells[self.n - 1]

    def observe(self, cell, result):
        pass


def _ship(*names, size=4):
    mask = 0
    for n in names:
        mask |= 1 << parse_cell(n, size)
    return mask


def test_first_player_wins_a_tie_in_shots():
    a_fleet = (_ship("A1", "A2"),)
    b_fleet = (_ship("D3", "D4"),)
    # Both need exactly 2 shots; whoever goes first sinks first.
    for first in (0, 1):
        r = race((a_fleet, b_fleet), (Scripted(["D3", "D4"]), Scripted(["A1", "A2"])), TINY,
                 first=first)
        assert r.winner == first
        assert r.turns == 3
        assert r.shots == ((2, 1) if first == 0 else (1, 2))


def test_salvo_fires_one_shot_per_ship_afloat():
    rules = Rules(size=4, fleet=(Ship("A", 2), Ship("B", 2)), salvo=True)
    a_fleet = (_ship("A1", "A2"), _ship("C1", "C2"))
    b_fleet = (_ship("A3", "A4"), _ship("D3", "D4"))
    a = Scripted(["B1", "B2", "A3", "A4", "D3", "D4"])
    b = Scripted(["A1", "A2", "B3", "B4", "C3", "C4"])
    r = race((a_fleet, b_fleet), (a, b), rules, first=0)
    # T1 A fires 2 (B1 B2 miss). T2 B fires 2 (A1 A2: sinks A's first ship).
    # T3 A, one ship left, fires 1 (A3 hit). T4 B fires 2 (B3 B4 miss).
    # T5 A fires A4 (sinks B's first). T6 B, one ship left, fires 1 (C3 miss).
    # T7 A fires D3. T8 B fires C4. T9 A fires D4 and wins.
    assert (r.winner, r.turns, r.shots) == (0, 9, (6, 6))
    assert r.afloat == (1, 0)
    assert a.n == r.shots[0] and b.n == r.shots[1]
    assert Rules(salvo=True).describe()["salvo"] is True
    assert "salvo" not in Rules().describe()  # keeps older lab fingerprints stable


def test_salvo_volley_sizes():
    rules = Rules(size=4, fleet=(Ship("A", 1), Ship("B", 1), Ship("C", 1)), salvo=True)
    a_fleet = (_ship("A1"), _ship("A2"), _ship("A3"))
    b_fleet = (_ship("D1"), _ship("D2"), _ship("D3"))
    a = Scripted(["B1", "B2", "B3", "B4", "C1", "C2", "C3", "C4", "D1", "D2", "D3"])
    b = Scripted(["A1", "C1", "C2", "A2", "A3"])
    r = race((a_fleet, b_fleet), (a, b), rules, first=0)
    # A fires 3 misses; B sinks A1 with its first of 3; A now fires 2; B sinks A2, A3...
    assert r.winner == 1
    assert r.shots == (5, 5)


def test_race_mirror_matches_are_fair_and_seats_alternate():
    tally = RaceTally()
    for _, r in run_races(RaceConfig(("hunt-target", "hunt-target"), games=400, seed="m",
                                     workers=1)):
        tally.add(r)
    assert tally.went_first == [200, 200]
    s = tally.summary(("a", "b"))
    assert s["players"][0]["ci95"][0] < 0.5 < s["players"][0]["ci95"][1]


def test_race_is_reproducible_across_workers():
    one = list(run_races(RaceConfig(("random", "hunt-target"), games=60, seed="r", workers=1,
                                    chunk_size=7)))
    two = list(run_races(RaceConfig(("random", "hunt-target"), games=60, seed="r", workers=2,
                                    chunk_size=7)))
    assert one == two


def test_first_mover_exact_formula_small_case():
    # One 1-cell ship on 2 cells: each needs 1 or 2 shots with equal chance.
    # First wins unless it needs 2 and the other needs 1: 1 - 1/4.
    assert exact.first_player_wins_random(1, 2) == pytest.approx(0.75)


# -- learning ---------------------------------------------------------------------


def test_opponent_aware_starts_as_a_plain_probability_map():
    rules = Rules()
    learner = OpponentAware(rules, random.Random(1))
    plain = make_shooter("probability-map", rules, random.Random(1))
    assert learner.scores() == plain.scores()


def test_opponent_aware_learns_from_revealed_fleets():
    rules = Rules()
    learner = OpponentAware(rules, random.Random(0))
    edge = 0
    for c in range(10):
        edge |= 1 << c  # row A
    for _ in range(10):
        learner.knowledge.hits = edge  # pretend each game's ships were all on row A
        learner.new_game()
    assert learner.games_seen == 10
    scores = learner.scores()
    assert max(scores[:10]) > max(scores[40:60])
    assert learner.knowledge.hits == 0  # fresh knowledge for the new game


def test_series_learning_shooter_keeps_memory_and_others_restart():
    config = SeriesConfig(("opponent-aware", "probability-map"), placer="edges", series=1,
                          length=5, seed="s")
    result = play_series(config, 0)
    assert all(len(v) == 5 for v in result.values())
    assert all(17 <= x <= 100 for v in result.values() for x in v)


def test_series_tally_and_reproducibility():
    config = SeriesConfig(("probability-map",), placer="edges", series=6, length=3, seed="t",
                          workers=1, chunk_size=4)
    a = list(run_series(config))
    b = list(run_series(SeriesConfig(**{**config.__dict__, "workers": 2})))
    assert a == b
    tally = SeriesTally(3)
    for _, r in a:
        tally.add(r)
    s = tally.summary()
    assert s["series"] == 6 and len(s["shooters"]["probability-map"]) == 3
    mean = sum(r["probability-map"][0] for _, r in a) / 6
    assert s["shooters"]["probability-map"][0]["mean"] == pytest.approx(mean)


def test_cli_race_and_series(tmp_path, capsys):
    main(["race", "random", "hunt-target", "-n", "20", "--workers", "1", "--salvo",
          "--out", str(tmp_path / "race.json")])
    main(["series", "opponent-aware", "probability-map", "--series", "3", "--length", "3",
          "--workers", "1", "--out", str(tmp_path / "series.json")])
    out = capsys.readouterr().out
    assert "Salvo" in out and "game" in out
    assert (tmp_path / "race.json").exists() and (tmp_path / "series.json").exists()
