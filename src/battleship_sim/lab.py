"""Continuous experiment runner: plays batches forever and publishes live statistics.

Each condition (shooters against one placer under one ruleset) has its own seed, and its
games are numbered from 0, so any published number can be reproduced exactly with
`battleship-sim run`. Totals are exact running sums plus a histogram of shot counts, so
the published means, intervals, medians and percentiles match what `run` reports.

Every condition carries a behavioral fingerprint: a hash of a fixed set of canary games
played with its shooters, placer and rules. If a code change alters how that condition
plays, its fingerprint changes and its totals restart (the old ones are archived), while
conditions the change didn't touch keep counting. A change that affected none of the
canary games could slip through, but its effect on the totals would be tiny.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import signal
import subprocess
import time
import tomllib
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from battleship_sim import __version__
from battleship_sim.game import SolveResult
from battleship_sim.placers import get_placer
from battleship_sim.rules import FLEET, Feedback, Rules, parse_fleet
from battleship_sim.shooters import make_shooter
from battleship_sim.simulate import SimConfig, _play_chunk, exact_mean, ship_columns
from battleship_sim.stats import Z95

STATUS = "Live simulation output. Preliminary, not reviewed findings."
REPOSITORY = "https://github.com/kalebhoffer/battleship-strategy-sim"
CANARY_GAMES = 300
CDF_SHOTS = 100  # the published distribution covers 0..100 shots


@dataclass(frozen=True)
class Condition:
    id: str
    label: str
    study: str
    group: str  # conditions sharing a group are shown together as a shooter x placer table
    shooters: tuple[str, ...]
    placer: str
    rules: Rules

    def seed(self, base: str) -> str:
        return f"{base}/{self.id}"

    def definition(self) -> dict:
        return {
            "shooters": list(self.shooters),
            "placer": self.placer,
            "rules": self.rules.describe(),
        }


@dataclass(frozen=True)
class LabConfig:
    seed: str
    batch_games: int
    conditions: tuple[Condition, ...]


def _rules(raw: dict) -> Rules:
    return Rules(
        size=int(raw.get("size", 10)),
        fleet=parse_fleet(raw["fleet"]) if "fleet" in raw else FLEET,
        feedback=Feedback(raw.get("feedback", Feedback.NAMED_HITS.value)),
        touching=bool(raw.get("touching", True)),
    )


def load_config(path: Path) -> LabConfig:
    raw = tomllib.loads(path.read_text())
    conditions = []
    for c in raw["condition"]:
        cond = Condition(
            id=c["id"],
            label=c.get("label", c["id"]),
            study=str(c.get("study", "")),
            group=c.get("group", ""),
            shooters=tuple(dict.fromkeys(c["shooters"])),
            placer=c.get("placer", "uniform"),
            rules=_rules(c.get("rules", {})),
        )
        for name in cond.shooters:
            make_shooter(name, cond.rules, random.Random(0))
        get_placer(cond.placer)
        conditions.append(cond)
    ids = [c.id for c in conditions]
    if len(ids) != len(set(ids)):
        raise ValueError("condition ids must be unique")
    return LabConfig(
        str(raw.get("seed", "0")), int(raw.get("batch_games", 1000)), tuple(conditions)
    )


def fingerprint(cond: Condition) -> str:
    """Hash of the condition's definition and the outcome of fixed canary games."""
    sim = SimConfig(cond.shooters, cond.placer, CANARY_GAMES, f"canary/{cond.id}", cond.rules)
    h = hashlib.sha256(json.dumps(cond.definition(), sort_keys=True).encode())
    games, _ = _play_chunk(sim, 0, CANARY_GAMES)
    for _, results in games:
        for name, r in results:
            h.update(f"{name}:{r.shots}:{r.first_hit}:{r.sunk_at}:{r.repeats};".encode())
    return h.hexdigest()[:16]


# -- accumulated state -------------------------------------------------------


def new_condition_state(fp: str, ships: int) -> dict:
    return {"fingerprint": fp, "games": 0, "shooters": {}, "pairs": {}, "ships": ships}


def _shooter_state(ships: int) -> dict:
    return {
        "n": 0,
        "sum": 0,
        "sumsq": 0,
        "hist": {},  # shots -> games
        "capped": 0,
        "first_hit": [0, 0],  # [sum, n]
        "sunk": [[0, 0] for _ in range(ships)],
        "modes": {"hunt_shots": 0, "hunt_hits": 0, "target_shots": 0, "target_hits": 0},
        "has_modes": False,
    }


def add_game(cs: dict, results: list[tuple[str, SolveResult]]) -> None:
    cs["games"] += 1
    for name, r in results:
        s = cs["shooters"].setdefault(name, _shooter_state(cs["ships"]))
        s["n"] += 1
        s["sum"] += r.shots
        s["sumsq"] += r.shots * r.shots
        key = str(r.shots)
        s["hist"][key] = s["hist"].get(key, 0) + 1
        s["capped"] += r.capped
        if r.first_hit is not None:
            s["first_hit"][0] += r.first_hit
            s["first_hit"][1] += 1
        for i, at in enumerate(r.sunk_at):
            if at is not None:
                s["sunk"][i][0] += at
                s["sunk"][i][1] += 1
        if r.hunt_shots is not None:
            s["has_modes"] = True
            for k in s["modes"]:
                s["modes"][k] += getattr(r, k)
    shots = {name: r.shots for name, r in results}
    names = list(shots)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            p = cs["pairs"].setdefault(f"{a}|{b}", {"n": 0, "sum": 0, "sumsq": 0})
            d = shots[a] - shots[b]
            p["n"] += 1
            p["sum"] += d
            p["sumsq"] += d * d


def _moments(n: int, total: int, sumsq: int) -> tuple[float, float, float, float]:
    """Mean, sample sd, and a normal 95% interval, from exact integer sums."""
    mean = total / n
    var = (sumsq - total * total / n) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(max(0.0, var))
    half = Z95 * sd / math.sqrt(n) if n > 1 else 0.0
    return mean, sd, mean - half, mean + half


def _order_stat(hist: list[tuple[int, int]], k: int) -> int:
    """The k-th smallest value (0-based) of the histogram."""
    seen = 0
    for value, count in hist:
        seen += count
        if k < seen:
            return value
    raise IndexError(k)


def hist_quantile(hist: dict[str, int], q: float) -> float:
    """Same linear interpolation as stats.quantile, from a histogram."""
    items = sorted((int(v), c) for v, c in hist.items())
    n = sum(c for _, c in items)
    pos = q * (n - 1)
    low = math.floor(pos)
    high = min(low + 1, n - 1)
    a, b = _order_stat(items, low), _order_stat(items, high)
    return a + (b - a) * (pos - low)


def _cdf(hist: dict[str, int], n: int) -> list[float]:
    counts = [0] * (CDF_SHOTS + 1)
    for v, c in hist.items():
        if int(v) <= CDF_SHOTS:
            counts[int(v)] += c
    out, running = [], 0
    for c in counts:
        running += c
        out.append(round(running / n, 5))
    return out


def snapshot(config: LabConfig, state: dict, commit: str | None) -> dict:
    """The public JSON: computed statistics, plus everything needed to reproduce them."""
    conditions = []
    for cond in config.conditions:
        cs = state["conditions"].get(cond.id)
        shooters = []
        if cs:
            columns = ship_columns(cond.rules)
            for name in cond.shooters:
                s = cs["shooters"].get(name)
                if not s or not s["n"]:
                    continue
                mean, sd, low, high = _moments(s["n"], s["sum"], s["sumsq"])
                entry = {
                    "name": name,
                    "games": s["n"],
                    "mean": mean,
                    "ci95": [low, high],
                    "sd": sd,
                    "median": hist_quantile(s["hist"], 0.5),
                    "p5": hist_quantile(s["hist"], 0.05),
                    "p95": hist_quantile(s["hist"], 0.95),
                    "min": min(int(v) for v in s["hist"]),
                    "max": max(int(v) for v in s["hist"]),
                    "capped": s["capped"],
                    "exact_mean": exact_mean(name, cond.rules),
                    "mean_first_hit": (
                        s["first_hit"][0] / s["first_hit"][1] if s["first_hit"][1] else None
                    ),
                    "mean_sunk_at": {
                        col: (total / k if k else None)
                        for col, (total, k) in zip(columns, s["sunk"], strict=True)
                    },
                    "cdf": _cdf(s["hist"], s["n"]),
                }
                if s["has_modes"]:
                    m = s["modes"]
                    entry["hunt_hit_rate"] = (
                        m["hunt_hits"] / m["hunt_shots"] if m["hunt_shots"] else None
                    )
                    entry["target_hit_rate"] = (
                        m["target_hits"] / m["target_shots"] if m["target_shots"] else None
                    )
                    entry["share_of_shots_targeting"] = m["target_shots"] / s["sum"]
                shooters.append(entry)
        pairs = []
        for key, p in (cs or {}).get("pairs", {}).items():
            a, b = key.split("|")
            mean, _, low, high = _moments(p["n"], p["sum"], p["sumsq"])
            pairs.append({"a": a, "b": b, "games": p["n"], "mean_a_minus_b": mean,
                          "ci95": [low, high]})
        conditions.append(
            {
                "id": cond.id,
                "label": cond.label,
                "study": cond.study,
                "group": cond.group,
                "placer": cond.placer,
                "rules": cond.rules.describe(),
                "seed": cond.seed(config.seed),
                "fingerprint": cs["fingerprint"] if cs else None,
                "games": cs["games"] if cs else 0,
                "shooters": shooters,
                "paired": pairs,
                "reproduce": reproduce_command(cond, config.seed, cs["games"] if cs else 0),
            }
        )
    return {
        "status": STATUS,
        "generated_at": _now(),
        "started_at": state["started_at"],
        "total_games": sum(c["games"] for c in conditions),
        "engine": {"version": __version__, "commit": commit},
        "repository": REPOSITORY,
        "conditions": conditions,
    }


def reproduce_command(cond: Condition, seed: str, games: int) -> str:
    official = Rules()
    flags = []
    if cond.placer != "uniform":
        flags.append(f"--placer {cond.placer}")
    if cond.rules.feedback is not official.feedback:
        flags.append(f"--feedback {cond.rules.feedback}")
    if not cond.rules.touching:
        flags.append("--no-touching")
    if cond.rules.size != official.size:
        flags.append(f"--size {cond.rules.size}")
    if cond.rules.fleet != official.fleet:
        flags.append("--fleet " + ",".join(str(s.length) for s in cond.rules.fleet))
    return " ".join(
        ["battleship-sim run", *cond.shooters, f"-n {games}", f"--seed '{cond.seed(seed)}'",
         *flags]
    )


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, data: dict, indent: int | None = None) -> None:
    """Atomic write, so readers never see a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=indent, separators=None if indent else (",", ":"))
                   + "\n")
    os.replace(tmp, path)


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {"started_at": _now(), "conditions": {}, "retired": {}}


def sync_fingerprints(config: LabConfig, state: dict, log=print) -> None:
    """Retire any condition whose behavior changed; start missing ones from zero."""
    for cond in config.conditions:
        fp = fingerprint(cond)
        cs = state["conditions"].get(cond.id)
        if cs and cs["fingerprint"] != fp:
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            state.setdefault("retired", {})[f"{cond.id}@{cs['fingerprint']}@{stamp}"] = cs
            log(f"lab: {cond.id} plays differently now ({cs['fingerprint']} -> {fp}); "
                f"archived {cs['games']:,} games and restarting it")
            cs = None
        if cs is None:
            state["conditions"][cond.id] = new_condition_state(fp, len(cond.rules.fleet))


# -- main loop ---------------------------------------------------------------


def git_commit(cwd: Path) -> tuple[str | None, bool]:
    """HEAD commit, and whether the package source has uncommitted changes."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=cwd,
                               capture_output=True, text=True, check=True).stdout.strip()
        return head, bool(dirty)
    except (OSError, subprocess.CalledProcessError):
        return None, False


def upload(path: Path, dest: str, log=print) -> None:
    """Copy the snapshot to the web server. rsync writes a temp file and renames it."""
    try:
        subprocess.run(
            ["rsync", "-q", "--chmod=F644", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=15",
             str(path), dest],
            check=True, timeout=60, capture_output=True,
        )
    except (OSError, subprocess.SubprocessError) as e:
        log(f"lab: upload failed, will retry next publish ({e})")


def run_lab(
    config_path: Path,
    state_dir: Path,
    publish_path: Path,
    workers: int = 1,
    publish_every: float = 60.0,
    max_batches: int | None = None,
    upload_to: str | None = None,
    commit: str | None = None,
    log=print,
) -> dict:
    config = load_config(config_path)
    state_path = state_dir / "state.json"
    state = load_state(state_path)
    sync_fingerprints(config, state, log)
    stop = False

    def _stop(*_):
        nonlocal stop
        stop = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _stop)

    def publish() -> None:
        _write_json(publish_path, snapshot(config, state, commit))
        if upload_to:
            upload(publish_path, upload_to, log)

    log(f"lab: {len(config.conditions)} conditions, {workers} workers, commit {commit}")
    last_publish = 0.0
    batches = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while not stop and (max_batches is None or batches < max_batches):
            for cond in config.conditions:
                cs = state["conditions"][cond.id]
                for _, results in _run_batch(pool, workers, config, cond, cs["games"]):
                    add_game(cs, results)
                if stop:
                    break
            batches += 1
            _write_json(state_path, state)
            if time.monotonic() - last_publish >= publish_every or stop:
                publish()
                last_publish = time.monotonic()
                total = sum(c["games"] for c in state["conditions"].values())
                log(f"lab: {total:,} games published")
    _write_json(state_path, state)
    publish()
    return state


def _run_batch(
    pool: Executor, workers: int, config: LabConfig, cond: Condition, start: int
) -> list[tuple[int, list[tuple[str, SolveResult]]]]:
    stop = start + config.batch_games
    sim = SimConfig(cond.shooters, cond.placer, stop, cond.seed(config.seed), cond.rules)
    size = max(1, math.ceil(config.batch_games / workers))
    bounds = [(s, min(s + size, stop)) for s in range(start, stop, size)]
    futures = [pool.submit(_play_chunk, sim, a, b) for a, b in bounds]
    return [g for f in futures for g in f.result()[0]]
