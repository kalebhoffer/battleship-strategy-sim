"""Continuous experiment runner: plays batches forever and publishes live statistics.

Each condition (shooters against one placer under one ruleset) has its own seed, and its
games are numbered from 0, so any published number can be reproduced exactly with
`battleship-sim run`. Totals are exact running sums plus a histogram of shot counts, so
the published means, intervals, medians and percentiles match what `run` reports.

Every condition carries a behavioral fingerprint: a hash of a fixed set of canary games
played with its shooters, placer and rules. If a code change alters how that condition
plays, its fingerprint changes and its totals restart (the old ones are archived), while
conditions the change didn't touch keep counting. A change that affects only rare games
can miss the canaries, so after changing shooter code, replay a condition's first games
under the old and new code (`battleship-sim run` with its seed) and compare. If any game
differs, bump the condition's `revision` in live.toml to restart it.

A condition stops once it reaches its target (the research plan's stopping rule): a 95%
interval on every shooter's mean shot count (for a series, at every game number) narrower
than `target_ci`, or `target_games` games. Its workers then go to the conditions still
running. Targets are checked against the saved totals, so raising one in live.toml
resumes the condition where it stopped.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import signal
import subprocess
import threading
import time
import tomllib
from collections import deque
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from battleship_sim import __version__
from battleship_sim.game import SolveResult
from battleship_sim.placers import get_placer
from battleship_sim.race import RaceConfig, RaceTally
from battleship_sim.race import _chunk as race_chunk
from battleship_sim.rules import FLEET, Feedback, Rules, parse_fleet
from battleship_sim.series import SeriesConfig, SeriesTally
from battleship_sim.series import _chunk as series_chunk
from battleship_sim.shooters import make_shooter
from battleship_sim.simulate import SimConfig, _play_chunk, exact_mean, ship_columns
from battleship_sim.stats import Z95

STATUS = "Live simulation output. Preliminary, not reviewed findings."
REPOSITORY = "https://github.com/kalebhoffer/battleship-strategy-sim"
CANARY_GAMES = 300
CANARY_SERIES = 10
MAX_BATCH = 50_000
PIPELINE_DEPTH = 2  # batches a slow condition keeps queued
MIN_TARGET_GAMES = 100  # never call an interval "narrow enough" on fewer games than this
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
    # "solve" (each shooter alone against a fleet), "race" (two players take turns), or
    # "series" (many games against one opponent, keeping what was learned).
    kind: str = "solve"
    placers: tuple[str, str] = ("uniform", "uniform")  # race: how A and B hide ships
    length: int = 20  # series: games per opponent
    batch: int | None = None  # games (or series) per batch, overriding the lab default
    # Bump to restart a condition after a code change its canary games don't reveal.
    revision: int = 0
    # "slow" conditions run in their own lane with their own workers (scheduling only).
    lane: str = "fast"
    # When to stop (see the module docstring). Neither is part of the fingerprint.
    target_games: int | None = None
    target_ci: float | None = None  # overrides the lab's target_ci

    def seed(self, base: str) -> str:
        return f"{base}/{self.id}"

    def definition(self) -> dict:
        d: dict = {
            "shooters": list(self.shooters),
            "placer": self.placer,
            "rules": self.rules.describe(),
        }
        # Solve conditions keep the original shape so their fingerprints don't change.
        if self.kind == "race":
            d.update(kind="race", placers=list(self.placers))
        elif self.kind == "series":
            d.update(kind="series", length=self.length)
        if self.revision:
            d["revision"] = self.revision
        return d


@dataclass(frozen=True)
class LabConfig:
    seed: str
    batch_games: int  # first batch per condition; later batches are sized by time
    conditions: tuple[Condition, ...]
    batch_seconds: float = 3.0  # target wall time per batch; 0 = fixed-size batches
    target_ci: float | None = None  # solve conditions stop once every interval is this narrow
    race_games: int | None = None  # race conditions stop at this many games


def _rules(raw: dict) -> Rules:
    return Rules(
        size=int(raw.get("size", 10)),
        fleet=parse_fleet(raw["fleet"]) if "fleet" in raw else FLEET,
        feedback=Feedback(raw.get("feedback", Feedback.NAMED_HITS.value)),
        touching=bool(raw.get("touching", True)),
        salvo=bool(raw.get("salvo", False)),
    )


def load_config(path: Path) -> LabConfig:
    raw = tomllib.loads(path.read_text())
    conditions = []
    for c in raw["condition"]:
        kind = c.get("kind", "solve")
        if kind not in ("solve", "race", "series"):
            raise ValueError(f"{c['id']}: kind must be solve, race, or series")
        placer = c.get("placer", "uniform")
        shooters = tuple(c["shooters"])
        if kind == "race" and len(shooters) != 2:
            raise ValueError(f"{c['id']}: a race needs exactly two shooters")
        cond = Condition(
            id=c["id"],
            label=c.get("label", c["id"]),
            study=str(c.get("study", "")),
            group=c.get("group", ""),
            # A race may pit a shooter against itself; the other kinds list each once.
            shooters=shooters if kind == "race" else tuple(dict.fromkeys(shooters)),
            placer=placer,
            rules=_rules(c.get("rules", {})),
            kind=kind,
            placers=tuple(c.get("placers", [placer, placer])),
            length=int(c.get("length", 20)),
            batch=c.get("batch"),
            revision=int(c.get("revision", 0)),
            lane=c.get("lane", "fast"),
            target_games=c.get("target_games"),
            target_ci=c.get("target_ci"),
        )
        if cond.lane not in ("fast", "slow"):
            raise ValueError(f"{cond.id}: lane must be fast or slow")
        for name in cond.shooters:
            make_shooter(name, cond.rules, random.Random(0))
        for name in {cond.placer, *cond.placers}:
            get_placer(name)
        conditions.append(cond)
    ids = [c.id for c in conditions]
    if len(ids) != len(set(ids)):
        raise ValueError("condition ids must be unique")
    return LabConfig(
        str(raw.get("seed", "0")), int(raw.get("batch_games", 1000)), tuple(conditions),
        float(raw.get("batch_seconds", 3.0)), raw.get("target_ci"), raw.get("race_games"),
    )


def fingerprint(cond: Condition) -> str:
    """Hash of the condition's definition and the outcome of fixed canary games."""
    h = hashlib.sha256(json.dumps(cond.definition(), sort_keys=True).encode())
    seed = f"canary/{cond.id}"
    # Slow conditions set a small batch; their canary is no bigger than one batch.
    games = min(CANARY_GAMES, cond.batch) if cond.batch else CANARY_GAMES
    if cond.kind == "race":
        races = race_chunk(_race_config(cond, seed), 0, games)
        for _, r in races:
            h.update(f"{r.winner}:{r.turns}:{r.shots}:{r.afloat};".encode())
        return h.hexdigest()[:16]
    if cond.kind == "series":
        count = min(CANARY_SERIES, cond.batch) if cond.batch else CANARY_SERIES
        for _, result in series_chunk(_series_config(cond, seed), 0, count):
            h.update(json.dumps(result, sort_keys=True).encode())
        return h.hexdigest()[:16]
    sim = SimConfig(cond.shooters, cond.placer, games, seed, cond.rules)
    played, _ = _play_chunk(sim, 0, games)
    for _, results in played:
        for name, r in results:
            h.update(f"{name}:{r.shots}:{r.first_hit}:{r.sunk_at}:{r.repeats};".encode())
    return h.hexdigest()[:16]


# -- accumulated state -------------------------------------------------------


def _race_config(cond: Condition, seed: str, games: int = 0) -> RaceConfig:
    return RaceConfig((cond.shooters[0], cond.shooters[1]), cond.placers, games, seed,
                      cond.rules, workers=1)


def _series_config(cond: Condition, seed: str, series: int = 0) -> SeriesConfig:
    return SeriesConfig(cond.shooters, cond.placer, series, cond.length, seed, cond.rules,
                        workers=1)


def new_condition_state(fp: str, cond: Condition) -> dict:
    cs = {"fingerprint": fp, "games": 0, "shooters": {}, "pairs": {},
          "ships": len(cond.rules.fleet)}
    if cond.kind != "solve":
        cs["kind"] = cond.kind
    if cond.kind == "race":
        cs["race"] = asdict(RaceTally())
    elif cond.kind == "series":
        cs["series"] = asdict(SeriesTally(cond.length))
    return cs


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


def target_of(config: LabConfig, cond: Condition) -> dict:
    """The condition's stopping target: {"games": n} and/or {"ci95_width": w}."""
    target: dict = {}
    games = cond.target_games or (config.race_games if cond.kind == "race" else None)
    if games:
        target["games"] = games
    # A condition's own target_games replaces the lab-wide interval target.
    ci = cond.target_ci if cond.target_ci is not None or cond.target_games else config.target_ci
    if cond.kind != "race" and ci:
        target["ci95_width"] = ci
    return target


def widest_interval(cs: dict) -> float | None:
    """Width of the widest 95% interval on a mean shot count: across the shooters of a
    solve condition, or across every shooter and game number of a series condition."""
    if "series" in cs:
        n = cs["series"]["series"]
        cells = [(n, total, sumsq) for rows in cs["series"]["sums"].values()
                 for total, sumsq in rows]
    else:
        cells = [(s["n"], s["sum"], s["sumsq"]) for s in cs["shooters"].values()]
    if not cells or min(c[0] for c in cells) < MIN_TARGET_GAMES:
        return None
    return max(high - low for _, _, low, high in (_moments(*c) for c in cells))


def reached_target(config: LabConfig, cond: Condition, cs: dict | None) -> bool:
    """Whether the condition has met either of its targets (none set: never)."""
    if not cs:
        return False
    target = target_of(config, cond)
    if "games" in target and cs["games"] >= target["games"]:
        return True
    if "ci95_width" in target:
        width = widest_interval(cs)
        return width is not None and width < target["ci95_width"]
    return False


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
        extra: dict = {"kind": cond.kind}
        if cs and cond.kind == "race":
            extra["race"] = RaceTally(**cs["race"]).summary((cond.shooters[0], cond.shooters[1]))
            extra["placers"] = list(cond.placers)
        elif cs and cond.kind == "series":
            extra["series"] = SeriesTally(**cs["series"]).summary()
        conditions.append(
            {
                **extra,
                "id": cond.id,
                "label": cond.label,
                "study": cond.study,
                "group": cond.group,
                "placer": cond.placer,
                "rules": cond.rules.describe(),
                "seed": cond.seed(config.seed),
                "fingerprint": cs["fingerprint"] if cs else None,
                "games": cs["games"] if cs else 0,
                "target": target_of(config, cond),
                "done": reached_target(config, cond, cs),
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
    flags = _rule_flags(cond.rules)
    if cond.kind == "race":
        a, b = cond.shooters
        return " ".join(["battleship-sim race", a, b, f"--placer-a {cond.placers[0]}",
                         f"--placer-b {cond.placers[1]}", f"-n {games}",
                         f"--seed '{cond.seed(seed)}'", *flags])
    if cond.kind == "series":
        return " ".join(["battleship-sim series", *cond.shooters, f"--placer {cond.placer}",
                         f"--series {games // cond.length}", f"--length {cond.length}",
                         f"--seed '{cond.seed(seed)}'", *flags])
    if cond.placer != "uniform":
        flags.insert(0, f"--placer {cond.placer}")
    return " ".join(
        ["battleship-sim run", *cond.shooters, f"-n {games}", f"--seed '{cond.seed(seed)}'",
         *flags]
    )


def _rule_flags(rules: Rules) -> list[str]:
    official = Rules()
    flags = []
    if rules.salvo:
        flags.append("--salvo")
    if rules.feedback is not official.feedback:
        flags.append(f"--feedback {rules.feedback}")
    if not rules.touching:
        flags.append("--no-touching")
    if rules.size != official.size:
        flags.append(f"--size {rules.size}")
    if rules.fleet != official.fleet:
        flags.append("--fleet " + ",".join(str(s.length) for s in rules.fleet))
    return flags


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


def sync_fingerprints(config: LabConfig, state: dict, log=print, pool: Executor | None = None
                      ) -> None:
    """Retire any condition whose behavior changed; start missing ones from zero."""
    if pool is None:
        prints = [fingerprint(c) for c in config.conditions]
    else:
        prints = list(pool.map(fingerprint, config.conditions))
    for cond, fp in zip(config.conditions, prints, strict=True):
        cs = state["conditions"].get(cond.id)
        if cs and cs["fingerprint"] != fp:
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            state.setdefault("retired", {})[f"{cond.id}@{cs['fingerprint']}@{stamp}"] = cs
            log(f"lab: {cond.id} plays differently now ({cs['fingerprint']} -> {fp}); "
                f"archived {cs['games']:,} games and restarting it")
            cs = None
        if cs is None:
            state["conditions"][cond.id] = new_condition_state(fp, cond)


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
    """Copy the snapshot to the web server. rsync writes a temp file and renames it.

    Only options macOS's openrsync also supports: -p keeps the file's 644 mode.
    """
    path.chmod(0o644)
    try:
        subprocess.run(
            ["rsync", "-q", "-p", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=15",
             str(path), dest],
            check=True, timeout=60, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        log(f"lab: upload failed, will retry next publish: {(e.stderr or '').strip()}")
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
    slow_workers: int | None = None,
) -> dict:
    """Play every condition until it reaches its target, in two lanes side by side.

    Conditions marked `lane = "slow"` (the Bayesian shooters under the harder rules) get
    `slow_workers` of the workers and the rest share the others, so the fast conditions
    never wait for a slow batch. Within a lane, conditions take turns, one batch at a
    time, so each condition's games stay numbered 0, 1, 2, ... with no gaps. A condition
    that reaches its target drops out, and once a whole lane is done the other lane gets
    every worker. When everything is done the lab idles until it is stopped.
    """
    config = load_config(config_path)
    state_path = state_dir / "state.json"
    state = load_state(state_path)
    lock = threading.Lock()
    stop = threading.Event()
    pace: dict[str, float] = {}  # games (series, for series conditions) per second

    def _stop(*_):
        stop.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _stop)

    lanes = {
        k: [c for c in config.conditions if (c.lane == "slow") == (k == "slow")]
        for k in ("fast", "slow")
    }
    lanes = {k: v for k, v in lanes.items() if v}
    if len(lanes) == 2:
        slow = slow_workers if slow_workers is not None else max(1, workers // 4)
        slow = min(max(1, slow), max(1, workers - 1))
        sizes = {"fast": max(1, workers - slow), "slow": slow}
    else:
        sizes = dict.fromkeys(lanes, workers)
    # The fast lane is one thread taking turns through its conditions. Each slow condition
    # gets a thread of its own: while one waits on its batch's longest game, the others'
    # games keep the idle workers busy.
    groups = [("fast", lanes["fast"])] if "fast" in lanes else []
    groups += [("slow", [c]) for c in lanes.get("slow", [])]
    running = {k: sum(1 for lane, _ in groups if lane == k) for k in lanes}
    finished: set[str] = set()

    def share(lane: str) -> int:
        """Workers this lane may use now: all of them once the other lane is done."""
        with lock:
            alone = sum(1 for v in running.values() if v) < 2
        return workers if alone else sizes[lane]

    def publish() -> None:
        with lock:
            snap = snapshot(config, state, commit)
            _write_json(state_path, state)
            total = sum(c["games"] for c in state["conditions"].values())
        _write_json(publish_path, snap)
        if upload_to:
            upload(publish_path, upload_to, log)
        log(f"lab: {total:,} games published")

    def still_running(conditions: list[Condition]) -> list[Condition]:
        out = []
        for cond in conditions:
            with lock:
                done = reached_target(config, cond, state["conditions"][cond.id])
            if not done:
                out.append(cond)
            elif cond.id not in finished:
                finished.add(cond.id)
                games = state["conditions"][cond.id]["games"]
                log(f"lab: {cond.id} reached its target at {games:,} games; stopping it")
        return out

    def run_lane(lane: str, conditions: list[Condition]) -> None:
        rounds = 0
        try:
            if lane == "slow":
                (cond,) = conditions
                play_pipelined(lane, cond)
                return
            while not stop.is_set() and (max_batches is None or rounds < max_batches):
                todo = still_running(conditions)
                if not todo:
                    break
                for cond in todo:
                    if stop.is_set():
                        break
                    play_batch(cond, share(lane))
                rounds += 1
        finally:
            with lock:
                running[lane] -= 1
                lane_done = running[lane] == 0
            if lane_done and not stop.is_set() and not still_running(lanes[lane]):
                log(f"lab: every {lane} condition reached its target")

    def next_game(cond: Condition) -> int:
        # Only this condition's thread touches it, so reading its start is safe unlocked.
        cs = state["conditions"][cond.id]
        if cond.kind == "race":
            return cs["race"]["games"]
        if cond.kind == "series":
            return cs["series"]["series"]
        return cs["games"]

    def batch_size(cond: Condition, n: int) -> int:
        # Size each batch to take about batch_seconds, from this condition's recent pace,
        # so cheap conditions play many games per turn and expensive ones few, rounded up
        # to a multiple of the workers so none sits idle. Batch size never affects
        # results: games are numbered and always played in order.
        if cond.id in pace and config.batch_seconds > 0:
            count = max(1, min(MAX_BATCH, round(pace[cond.id] * config.batch_seconds)))
            return math.ceil(count / n) * n
        return first_batch(config, cond)

    def note_pace(cond: Condition, count: int, seconds: float) -> None:
        rate = count / max(seconds, 1e-3)
        pace[cond.id] = rate if cond.id not in pace else (pace[cond.id] + rate) / 2

    def play_batch(cond: Condition, n: int) -> None:
        count = batch_size(cond, n)
        began = time.monotonic()
        played = _collect(cond, _submit_batch(pool, n, config, cond, next_game(cond), count))
        note_pace(cond, count, time.monotonic() - began)
        record(cond, played)

    def play_pipelined(lane: str, cond: Condition) -> None:
        """Keep the next batch queued while waiting for the current one, so workers that
        finish early start on it instead of waiting for the batch's longest game. Batches
        are still recorded strictly in order; any still queued when the condition stops
        are dropped, and the next run starts from the last recorded game."""
        queued: deque[tuple[int, list[Future]]] = deque()
        start = next_game(cond)
        began = time.monotonic()
        recorded = 0
        rounds = 0
        try:
            while not stop.is_set() and (max_batches is None or rounds < max_batches):
                if not still_running([cond]):
                    break
                while len(queued) < PIPELINE_DEPTH:
                    n = share(lane)
                    count = batch_size(cond, n)
                    queued.append((count, _submit_batch(pool, n, config, cond, start, count)))
                    start += count
                count, futures = queued.popleft()
                played = _collect(cond, futures)
                # Pace over the whole run: queued batches finish close together, so the
                # gap between two completions says nothing about speed.
                recorded += count
                pace[cond.id] = recorded / max(time.monotonic() - began, 1e-3)
                record(cond, played)
                rounds += 1
        finally:
            for _, futures in queued:
                for f in futures:
                    f.cancel()

    def record(cond: Condition, played: list) -> None:
        cs = state["conditions"][cond.id]
        with lock:
            if cond.kind == "race":
                tally = RaceTally(**cs["race"])
                for _, r in played:
                    tally.add(r)
                cs["race"] = asdict(tally)
                cs["games"] = tally.games
            elif cond.kind == "series":
                tally = SeriesTally(**cs["series"])
                for _, result in played:
                    tally.add(result)
                cs["series"] = asdict(tally)
                cs["games"] = tally.series * cond.length
            else:
                for _, results in played:
                    add_game(cs, results)

    lane_text = ", ".join(f"{len(lanes[k])} {k} on {sizes[k]}" for k in lanes)
    log(f"lab: {len(config.conditions)} conditions ({lane_text} workers), commit {commit}")
    # One pool for both lanes: each lane splits its batches into as many chunks as it
    # has workers, so together they never ask for more than the pool holds.
    pool = ProcessPoolExecutor(max_workers=workers)
    try:
        sync_fingerprints(config, state, log, pool)
        threads = [threading.Thread(target=run_lane, args=(lane, conds), daemon=True,
                                    name=f"lab-{lane}-{i}")
                   for i, (lane, conds) in enumerate(groups)]
        for t in threads:
            t.start()
        last_publish = time.monotonic()
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=1.0)
            if time.monotonic() - last_publish >= publish_every:
                publish()
                last_publish = time.monotonic()
    finally:
        pool.shutdown(cancel_futures=True)
    publish()
    if max_batches is None and not stop.is_set():
        # Exiting would only make launchd start the lab again; wait to be stopped.
        log("lab: all conditions reached their targets; idling")
        stop.wait()
    return state


def first_batch(config: LabConfig, cond: Condition) -> int:
    """Size of a condition's first batch: games, or series for a series condition."""
    if cond.kind == "series":
        return cond.batch or max(1, config.batch_games // cond.length)
    return cond.batch or config.batch_games


def _submit_batch(pool: Executor, workers: int, config: LabConfig, cond: Condition,
                  start: int, count: int) -> list[Future]:
    """Queue `count` games (races, series) numbered from `start`; see _collect."""
    if cond.kind == "solve":
        return _submit_solve_batch(pool, workers, config, cond, start, count)
    return _submit_kind_batch(pool, workers, config, cond, start, count)


def _collect(cond: Condition, futures: list[Future]) -> list:
    """Wait for a batch queued by _submit_batch and return its games in order."""
    if cond.kind == "solve":
        return [g for f in futures for g in f.result()[0]]
    return [x for f in futures for x in f.result()]


def _submit_kind_batch(pool: Executor, workers: int, config: LabConfig, cond: Condition,
                       start: int, count: int) -> list[Future]:
    """`count` races, or series, numbered from `start`."""
    if cond.kind == "race":
        cfg, chunk = _race_config(cond, cond.seed(config.seed)), race_chunk
    else:
        cfg, chunk = _series_config(cond, cond.seed(config.seed)), series_chunk
    stop = start + count
    size = max(1, math.ceil(count / workers))
    return [pool.submit(chunk, cfg, a, min(a + size, stop)) for a in range(start, stop, size)]


def _submit_solve_batch(
    pool: Executor, workers: int, config: LabConfig, cond: Condition, start: int, count: int
) -> list[Future]:
    stop = start + count
    sim = SimConfig(cond.shooters, cond.placer, stop, cond.seed(config.seed), cond.rules)
    size = max(1, math.ceil((stop - start) / workers))
    bounds = [(s, min(s + size, stop)) for s in range(start, stop, size)]
    return [pool.submit(_play_chunk, sim, a, b) for a, b in bounds]
