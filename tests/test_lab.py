import json
import shlex
from pathlib import Path

import pytest

from battleship_sim.cli import main
from battleship_sim.lab import fingerprint, hist_quantile, load_config, run_lab
from battleship_sim.rules import Rules
from battleship_sim.stats import quantile

CONFIG = """
seed = "t"
batch_games = 40

[[condition]]
id = "a"
study = "1"
label = "Random"
shooters = ["random", "hunt-target"]

[[condition]]
id = "b"
study = "3"
label = "Sink only, no touching"
shooters = ["hunt-target-parity", "probability-map"]
rules = { feedback = "sink-only", touching = false }
"""


@pytest.fixture
def lab(tmp_path):
    config = tmp_path / "live.toml"
    config.write_text(CONFIG)
    state = run_lab(config, tmp_path / "state", tmp_path / "pub.json", workers=1,
                    publish_every=0, max_batches=2, log=lambda _: None)
    return tmp_path, config, state, json.loads((tmp_path / "pub.json").read_text())


def test_snapshot_matches_an_offline_run(lab, tmp_path):
    """Every published statistic equals what `battleship-sim run` reports for the same
    games, using the snapshot's own reproduce command."""
    _, _, _, snap = lab
    for cond in snap["conditions"]:
        assert cond["games"] == 80
        out = tmp_path / f"replay-{cond['id']}"
        args = shlex.split(cond["reproduce"])[1:]  # drop "battleship-sim"
        main([*args, "--workers", "1", "--out", str(out)])
        summary = json.loads((out / "summary.json").read_text())
        for s in cond["shooters"]:
            offline = summary["shooters"][s["name"]]
            for key in ("games", "mean", "sd", "median", "p5", "p95", "min", "max", "capped",
                        "mean_first_hit", "mean_sunk_at", "hunt_hit_rate", "target_hit_rate",
                        "share_of_shots_targeting"):
                assert s.get(key) == pytest.approx(offline.get(key)), (cond["id"], key)
            assert s["ci95"] == pytest.approx(offline["ci95"])
        (pair,) = cond["paired"]
        (offline_pair,) = summary["paired"]
        assert pair["mean_a_minus_b"] == pytest.approx(offline_pair["mean_a_minus_b"])
        assert pair["ci95"] == pytest.approx(offline_pair["ci95"])


def test_snapshot_labels_and_cdf(lab):
    _, _, _, snap = lab
    assert "Preliminary" in snap["status"]
    assert snap["total_games"] == 160
    cond = snap["conditions"][0]
    assert cond["seed"] == "t/a"
    s = cond["shooters"][0]
    assert s["name"] == "random"
    assert len(s["cdf"]) == 101
    assert s["cdf"][16] == 0 and s["cdf"][100] == 1
    assert s["exact_mean"] == pytest.approx(17 * 101 / 18)
    assert "--feedback sink-only" in snap["conditions"][1]["reproduce"]
    assert "--no-touching" in snap["conditions"][1]["reproduce"]


def test_resume_continues_numbering(lab):
    tmp_path, config, _, _ = lab
    state = run_lab(config, tmp_path / "state", tmp_path / "pub.json", workers=1,
                    publish_every=0, max_batches=1, log=lambda _: None)
    assert state["conditions"]["a"]["games"] == 120
    assert not state["retired"]


def test_changed_condition_restarts_only_itself(lab):
    tmp_path, config, _, _ = lab
    config.write_text(CONFIG.replace('["random", "hunt-target"]', '["random", "pattern-rows"]'))
    logs = []
    state = run_lab(config, tmp_path / "state", tmp_path / "pub.json", workers=1,
                    publish_every=0, max_batches=1, log=logs.append)
    assert state["conditions"]["a"]["games"] == 40  # restarted
    assert state["conditions"]["b"]["games"] == 120  # untouched, kept counting
    assert any("archived 80 games" in line for line in logs)
    (retired,) = state["retired"].values()
    assert retired["games"] == 80


def test_fingerprint_tracks_behavior(monkeypatch, tmp_path):
    config = tmp_path / "live.toml"
    config.write_text(CONFIG)
    cond = load_config(config).conditions[0]
    before = fingerprint(cond)
    assert fingerprint(cond) == before
    # A behavior change in a shooter the condition uses changes its fingerprint.
    from battleship_sim.shooters import naive

    original = naive.RandomShooter.__init__

    def reversed_order(self, rules, rng):
        original(self, rules, rng)
        self._order.reverse()

    monkeypatch.setattr(naive.RandomShooter, "__init__", reversed_order)
    assert fingerprint(cond) != before


def test_hist_quantile_matches_sorted_quantile():
    values = [17, 40, 40, 41, 55, 60, 60, 60, 99, 100, 31]
    hist = {}
    for v in values:
        hist[str(v)] = hist.get(str(v), 0) + 1
    for q in (0.05, 0.25, 0.5, 0.9, 0.95):
        assert hist_quantile(hist, q) == pytest.approx(quantile(sorted(values), q))


def test_shipped_config_loads():
    config = load_config(Path(__file__).parents[1] / "experiments" / "live.toml")
    assert len({c.id for c in config.conditions}) == len(config.conditions)


def test_lab_refuses_dirty_source(monkeypatch):
    monkeypatch.setattr("battleship_sim.lab.git_commit", lambda _: ("abc", True))
    with pytest.raises(SystemExit):
        main(["lab", "--max-batches", "0"])


def test_upload_reports_rsync_errors(tmp_path):
    from battleship_sim.lab import upload

    snap = tmp_path / "x.json"
    snap.write_text("{}")
    logs = []
    upload(snap, str(tmp_path / "missing-dir" / "deeper" / ""), logs.append)
    assert logs and "upload failed" in logs[0]
    upload(snap, str(tmp_path / "out.json"), logs.append)
    assert (tmp_path / "out.json").read_text() == "{}"
    assert len(logs) == 1


KINDS = """
seed = "k"
batch_games = 30

[[condition]]
id = "solo"
shooters = ["hunt-target"]

[[condition]]
id = "duel"
kind = "race"
group = "Going first"
shooters = ["hunt-target", "hunt-target"]
rules = { salvo = true }

[[condition]]
id = "learn"
kind = "series"
group = "Learning"
placer = "edges"
length = 5
batch = 3
shooters = ["opponent-aware", "probability-map"]
"""


def test_race_and_series_conditions_match_offline_runs(tmp_path):
    config = tmp_path / "live.toml"
    config.write_text(KINDS)
    run_lab(config, tmp_path / "state", tmp_path / "pub.json", workers=2, publish_every=0,
            max_batches=2, log=lambda _: None)
    snap = json.loads((tmp_path / "pub.json").read_text())
    duel, learn = snap["conditions"][1], snap["conditions"][2]
    assert duel["kind"] == "race" and duel["games"] == 60
    assert "--salvo" in duel["reproduce"]
    assert learn["kind"] == "series" and learn["games"] == 30  # 6 series x 5 games

    out = tmp_path / "race.json"
    main([*shlex.split(duel["reproduce"])[1:], "--workers", "1", "--out", str(out)])
    offline = json.loads(out.read_text())
    for key in ("games", "first_player_win_rate", "mean_turns", "players"):
        assert duel["race"][key] == offline[key], key

    out = tmp_path / "series.json"
    main([*shlex.split(learn["reproduce"])[1:], "--workers", "1", "--out", str(out)])
    offline = json.loads(out.read_text())
    assert learn["series"]["shooters"] == offline["shooters"]
    assert learn["series"]["series"] == offline["series"] == 6


def test_old_solve_fingerprints_unchanged_by_new_fields(tmp_path):
    """Phase 4 added kinds and Salvo; solve conditions must hash exactly as before."""
    config = tmp_path / "live.toml"
    config.write_text(CONFIG)
    cond = load_config(config).conditions[0]
    assert cond.definition() == {
        "shooters": ["random", "hunt-target"],
        "placer": "uniform",
        "rules": Rules().describe(),
    }
    assert "salvo" not in Rules().describe() and "kind" not in cond.definition()
