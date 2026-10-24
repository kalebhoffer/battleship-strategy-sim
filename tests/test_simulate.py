import json
from dataclasses import replace

import pytest

from battleship_sim.cli import main
from battleship_sim.rules import Feedback, Rules
from battleship_sim.simulate import SimConfig, fleet_for, run, run_to_csv, summarize


def _results(config):
    return [(i, [(n, r) for n, r in rs]) for i, rs in run(config)]


def test_same_seed_same_games():
    config = SimConfig(shooters=("random", "random-repeat"), games=50, seed="s", workers=1)
    assert _results(config) == _results(config)


def test_worker_count_does_not_change_results():
    one = SimConfig(shooters=("random",), games=60, seed="w", workers=1, chunk_size=7)
    two = SimConfig(shooters=("random",), games=60, seed="w", workers=2, chunk_size=7)
    assert _results(one) == _results(two)


def test_every_shooter_faces_the_same_fleets():
    alone = SimConfig(shooters=("random",), games=20, seed="c")
    together = SimConfig(shooters=("random-repeat", "random"), games=20, seed="c")
    assert [fleet_for(alone, i) for i in range(20)] == [fleet_for(together, i) for i in range(20)]
    # And a shooter's own shots don't depend on who else is in the lineup.
    a = {i: dict(rs)["random"] for i, rs in run(replace(alone, workers=1))}
    b = {i: dict(rs)["random"] for i, rs in run(replace(together, workers=1))}
    assert a == b


def test_different_seeds_differ():
    a = SimConfig(shooters=("random",), games=5, seed="a")
    b = SimConfig(shooters=("random",), games=5, seed="b")
    assert [fleet_for(a, i) for i in range(5)] != [fleet_for(b, i) for i in range(5)]


def test_unknown_names_fail_fast():
    with pytest.raises(ValueError):
        list(run(SimConfig(shooters=("nope",), games=1)))
    with pytest.raises(ValueError):
        list(run(SimConfig(shooters=("random",), placer="nope", games=1)))


def test_csv_and_summary(tmp_path):
    rules = Rules(feedback=Feedback.SINK_ONLY)
    config = SimConfig(shooters=("random", "random-repeat"), games=200, seed="t", rules=rules,
                       workers=1)
    timing = {}
    path = run_to_csv(config, tmp_path / "games.csv", timing=timing)
    lines = path.read_text().splitlines()
    assert lines[0].startswith("game,shooter,shots,first_hit,repeats,capped,hunt_shots,")
    assert "sunk_carrier" in lines[0]
    assert "sunk_patrol_boat" in lines[0]
    assert len(lines) == 1 + 200 * 2
    assert set(timing) == {"random", "random-repeat"}

    summary = summarize(path, rules)
    random_stats = summary["shooters"]["random"]
    assert random_stats["games"] == 200
    assert 17 <= random_stats["min"] <= random_stats["median"] <= random_stats["max"] <= 100
    assert random_stats["exact_mean"] == pytest.approx(17 * 101 / 18)
    assert random_stats["mean_sunk_at"]["sunk_carrier"] is not None
    (pair,) = summary["paired"]
    assert pair["a"] == "random" and pair["b"] == "random-repeat"
    assert pair["mean_a_minus_b"] < 0


def test_cli_run_and_summarize(tmp_path, capsys):
    out = tmp_path / "run"
    main(["run", "random", "-n", "100", "--seed", "cli", "--workers", "1", "--out", str(out),
          "--no-touching", "--feedback", "hit-only"])
    meta = json.loads((out / "run.json").read_text())
    assert meta["config"]["rules"]["touching"] is False
    assert meta["config"]["rules"]["feedback"] == "hit-only"
    assert meta["config"]["seed"] == "cli"
    assert (out / "summary.json").exists()
    main(["summarize", str(out)])
    assert "random" in capsys.readouterr().out


def test_cli_list_count_show(capsys):
    main(["list"])
    main(["count", "--size", "5", "--fleet", "3,2"])
    main(["show", "--seed", "1"])
    out = capsys.readouterr().out
    assert "uniform" in out and "random" in out
    assert "legal layouts" in out
    assert out.count("C") >= 5
