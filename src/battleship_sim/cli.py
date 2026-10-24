"""Command line interface: `battleship-sim run|summarize|list|count|show|validate`."""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from battleship_sim import __version__
from battleship_sim.board import render
from battleship_sim.count import count_layouts, independent_positions
from battleship_sim.placers import PLACERS, get_placer
from battleship_sim.rules import FLEET, Feedback, Rules, Ship, parse_fleet
from battleship_sim.shooters import SHOOTERS
from battleship_sim.simulate import SimConfig, run_to_csv, summarize


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _progress(done: int, total: int) -> None:
    if done == total or done % max(1, total // 100) == 0:
        print(f"\r  {done:,}/{total:,} games", end="", file=sys.stderr, flush=True)
        if done == total:
            print(file=sys.stderr)


def print_summary(summary: dict) -> None:
    shooters = summary["shooters"]
    width = max(10, *(len(name) for name in shooters)) + 2
    print(
        f"\n{'shooter':<{width}}{'games':>9}{'mean':>9}  {'95% CI':<17}"
        f"{'median':>7}{'p5':>6}{'p95':>6}{'max':>6}{'exact':>9}"
    )
    for name, s in shooters.items():
        low, high = s["ci95"]
        known = f"{s['exact_mean']:.2f}" if "exact_mean" in s else "-"
        print(
            f"{name:<{width}}{s['games']:>9,}{s['mean']:>9.2f}  {f'{low:.2f} - {high:.2f}':<17}"
            f"{s['median']:>7.0f}{s['p5']:>6.0f}{s['p95']:>6.0f}{s['max']:>6}{known:>9}"
        )
        if s["capped"]:
            print(f"{'':<{width}}{s['capped']:,} games hit the shot cap")
    if summary["paired"]:
        print("\npaired on the same fleets (shots of a minus shots of b)")
        for d in summary["paired"]:
            low, high = d["ci95"]
            print(f"  {d['a']} - {d['b']}: {d['mean_a_minus_b']:+.2f}  ({low:+.2f} to {high:+.2f})")


def _rules_from(args: argparse.Namespace) -> Rules:
    return Rules(
        size=args.size,
        fleet=parse_fleet(args.fleet),
        feedback=Feedback(args.feedback),
        touching=not args.no_touching,
    )


def _rules_from_json(data: dict) -> Rules:
    return Rules(
        size=data["size"],
        fleet=tuple(Ship(s["name"], s["length"]) for s in data["fleet"]),
        feedback=Feedback(data["feedback"]),
        touching=data["touching"],
    )


def _add_rule_args(p: argparse.ArgumentParser) -> None:
    rules = p.add_argument_group("rules (defaults are the official rules)")
    rules.add_argument("--size", type=int, default=10, help="grid is size x size")
    rules.add_argument(
        "--fleet",
        default=",".join(str(s.length) for s in FLEET),
        help="ship lengths, e.g. 5,4,3,3,2",
    )
    rules.add_argument(
        "--feedback",
        choices=[f.value for f in Feedback],
        default=Feedback.NAMED_HITS.value,
        help="named-hits (rulebook), sink-only, or hit-only",
    )
    rules.add_argument(
        "--no-touching", action="store_true", help="ships may not touch, even diagonally"
    )


def cmd_run(args: argparse.Namespace) -> None:
    config = SimConfig(
        shooters=tuple(dict.fromkeys(args.shooters)),
        placer=args.placer,
        games=args.games,
        seed=args.seed,
        rules=_rules_from(args),
        workers=args.workers,
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else Path("results") / f"run-{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    timing: dict[str, float] = {}
    progress = _progress if sys.stderr.isatty() else None
    csv_path = run_to_csv(config, out / "games.csv", progress, timing)
    elapsed = time.perf_counter() - started
    summary = summarize(csv_path, config.rules)
    meta = {
        "config": {**asdict(config), "rules": config.rules.describe()},
        "version": __version__,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "started_utc": stamp,
        "seconds": round(elapsed, 1),
        # Wall-clock time is machine-dependent and not part of the reproducible results.
        "ms_per_game": {
            name: round(1000 * s / config.games, 4) for name, s in sorted(timing.items())
        },
    }
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print_summary(summary)
    print(f"\nPlacer: {config.placer}. Feedback: {config.rules.feedback}. "
          f"Touching: {'allowed' if config.rules.touching else 'not allowed'}.")
    print(f"Wrote {out}/ in {elapsed:.1f}s")


def cmd_summarize(args: argparse.Namespace) -> None:
    folder = Path(args.run)
    meta = json.loads((folder / "run.json").read_text())
    rules = _rules_from_json(meta["config"]["rules"])
    print_summary(summarize(folder / "games.csv", rules))


def cmd_list(_: argparse.Namespace) -> None:
    print("shooters")
    for spec in SHOOTERS.values():
        print(f"  {spec.name:<20}{spec.description}")
    print("\nplacers")
    for spec in PLACERS.values():
        print(f"  {spec.name:<20}{spec.description}")


def cmd_count(args: argparse.Namespace) -> None:
    rules = _rules_from(args)
    started = time.perf_counter()
    total = count_layouts(rules)
    positions = independent_positions(rules)
    print(f"{total:,} legal layouts ({total / positions:.4%} of {positions:,} position tuples)")
    print(f"counted in {time.perf_counter() - started:.1f}s")


def cmd_show(args: argparse.Namespace) -> None:
    rules = _rules_from(args)
    layout = get_placer(args.placer)(rules, random.Random(args.seed))
    print(render(layout, rules))


def cmd_validate(args: argparse.Namespace) -> None:
    from battleship_sim.validate import run_all

    failures = 0
    for check in run_all(games=args.games, samples=args.samples, full_counts=not args.quick):
        mark = "PASS" if check.passed else "FAIL"
        failures += not check.passed
        print(f"[{mark}] {check.name}\n       {check.detail}", flush=True)
    print("\nall checks passed" if not failures else f"\n{failures} check(s) failed")
    if failures:
        sys.exit(1)


def cmd_race(args: argparse.Namespace) -> None:
    from battleship_sim.race import RaceConfig, RaceTally, run_races

    rules = _rules_from(args)
    if args.salvo:
        rules = Rules(rules.size, rules.fleet, rules.feedback, rules.touching, salvo=True)
    config = RaceConfig(
        shooters=(args.a, args.b),
        placers=(args.placer_a, args.placer_b),
        games=args.games,
        seed=args.seed,
        rules=rules,
        workers=args.workers,
    )
    tally = RaceTally()
    for done, (_, result) in enumerate(run_races(config), start=1):
        tally.add(result)
        if sys.stderr.isatty():
            _progress(done, config.games)
    summary = tally.summary(config.shooters)
    print(f"\n{summary['games']:,} games{' (Salvo)' if rules.salvo else ''}, "
          f"each player first in half; {summary['mean_turns']:.1f} turns on average\n")
    for side, p in zip("AB", summary["players"], strict=True):
        low, high = p["ci95"]
        placer = config.placers["AB".index(side)]
        print(f"  {side} {p['name']} (hides: {placer}): wins {p['win_rate']:.1%} "
              f"({low:.1%} - {high:.1%}); going first {p['win_rate_going_first']:.1%}, "
              f"second {p['win_rate_going_second']:.1%}")
    low, high = summary["first_player_ci95"]
    print(f"\nwhoever fires first wins {summary['first_player_win_rate']:.1%} "
          f"({low:.1%} - {high:.1%})")
    if summary["capped"]:
        print(f"{summary['capped']:,} games hit the turn cap")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = {"config": {"shooters": list(config.shooters), "placers": list(config.placers),
                           "games": config.games, "seed": config.seed,
                           "rules": rules.describe()},
                "version": __version__, "git_commit": _git_commit()}
        out.write_text(json.dumps({**summary, **meta}, indent=2) + "\n")
        print(f"Wrote {out}")


def cmd_series(args: argparse.Namespace) -> None:
    from battleship_sim.series import SeriesConfig, SeriesTally, run_series

    config = SeriesConfig(
        shooters=tuple(dict.fromkeys(args.shooters)),
        placer=args.placer,
        series=args.series,
        length=args.length,
        seed=args.seed,
        rules=_rules_from(args),
        workers=args.workers,
    )
    tally = SeriesTally(config.length)
    for done, (_, result) in enumerate(run_series(config), start=1):
        tally.add(result)
        if sys.stderr.isatty():
            _progress(done, config.series)
    summary = tally.summary()
    print(f"\n{summary['series']:,} opponents hiding ships with '{config.placer}', "
          f"{config.length} games each: average shots by game number\n")
    width = max(len(n) for n in config.shooters) + 2
    games = sorted({0, 1, 2, 4, 9, config.length - 1} & set(range(config.length)))
    print(f"{'game':<{width}}" + "".join(f"{g + 1:>9}" for g in games))
    for name in config.shooters:
        rows = summary["shooters"][name]
        print(f"{name:<{width}}" + "".join(f"{rows[g]['mean']:>9.2f}" for g in games))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = {"config": {"shooters": list(config.shooters), "placer": config.placer,
                           "series": config.series, "length": config.length,
                           "seed": config.seed, "rules": config.rules.describe()},
                "version": __version__, "git_commit": _git_commit()}
        out.write_text(json.dumps({**summary, **meta}, indent=2) + "\n")
        print(f"\nWrote {out}")


def cmd_matrix(args: argparse.Namespace) -> None:
    from battleship_sim.matrix import run_matrix

    shooters = [s.strip() for s in args.shooters.split(",")]
    placers = [p.strip() for p in args.placers.split(",")]
    result = run_matrix(shooters, placers, args.games, args.seed, _rules_from(args), args.workers)
    width = max(len(s) for s in shooters) + 2
    col = max(12, *(len(p) + 2 for p in placers))
    print(f"\naverage shots to sink the fleet ({args.games:,} games per cell; "
          f"the placer wants more, the shooter fewer)\n")
    print(f"{'':<{width}}" + "".join(f"{p:>{col}}" for p in placers))
    for s in shooters:
        print(f"{s:<{width}}" + "".join(f"{result['cells'][s][p]['mean']:>{col}.2f}"
                                         for p in placers))
    eq = result["equilibrium"]
    low, high = eq["value_bounds"]
    print(f"\nequilibrium (zero-sum, approximate): value between {low:.2f} and {high:.2f} shots")
    print("  placer mix:  " + ", ".join(f"{k} {v:.1%}" for k, v in eq["placer_mix"].items()
                                       if v >= 0.005))
    print("  shooter mix: " + ", ".join(f"{k} {v:.1%}" for k, v in eq["shooter_mix"].items()
                                       if v >= 0.005))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = {"version": __version__, "git_commit": _git_commit()}
        out.write_text(json.dumps({**result, **meta}, indent=2) + "\n")
        print(f"\nWrote {out}")
    print("\nThe matrix uses sampled averages, so the equilibrium is only as precise as they are.")


def cmd_worst_case(args: argparse.Namespace) -> None:
    from battleship_sim.matrix import worst_case

    rules = _rules_from(args)
    result = worst_case(args.shooter, rules, args.steps, args.games, args.seed,
                        log=lambda m: print(m, file=sys.stderr, flush=True))
    print(render(tuple(result["layout"]), rules))
    c = result["check"]
    low, high = c["ci95"]
    print(f"\n{args.shooter}: {result['start_mean']:.2f} shots on the starting layout, "
          f"{result['search_mean']:.2f} on this one during the search;")
    print(f"re-checked on {c['games']} fresh games: {c['mean']:.2f} ({low:.2f} - {high:.2f}), "
          f"worst {c['max']}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({**result, "version": __version__,
                                   "git_commit": _git_commit()}, indent=2) + "\n")


def cmd_lab(args: argparse.Namespace) -> None:
    from battleship_sim.lab import git_commit, run_lab

    commit, dirty = git_commit(Path(__file__).parent)
    if dirty and not args.allow_dirty:
        sys.exit(
            "battleship-sim lab: the source has uncommitted changes, so published numbers "
            "could not be reproduced from a commit. Commit first, or pass --allow-dirty."
        )
    run_lab(
        Path(args.config),
        Path(args.state_dir),
        Path(args.publish),
        workers=args.workers,
        publish_every=args.publish_every,
        max_batches=args.max_batches,
        upload_to=args.upload,
        slow_workers=args.slow_workers,
        commit=None if dirty else commit,
        log=lambda msg: print(f"{datetime.now(UTC):%Y-%m-%d %H:%M:%S} {msg}", flush=True),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="battleship-sim", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("run", help="simulate games and write results")
    p.add_argument("shooters", nargs="+", help="shooting strategies to compare, see `list`")
    p.add_argument("--placer", default="uniform", help="how fleets are placed, see `list`")
    p.add_argument("-n", "--games", type=int, default=10_000)
    p.add_argument("--seed", default="0", help="any string; same seed, same fleets and shots")
    p.add_argument("--workers", type=int, default=0, help="processes (0 = all CPUs)")
    p.add_argument("--out", help="output directory (default results/run-<timestamp>)")
    _add_rule_args(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("summarize", help="print the summary of an existing run folder")
    p.add_argument("run", help="folder with games.csv and run.json")
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("list", help="list shooters and placers")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("count", help="count every legal fleet layout exactly")
    _add_rule_args(p)
    p.set_defaults(func=cmd_count)

    p = sub.add_parser("show", help="print one fleet layout")
    p.add_argument("--placer", default="uniform")
    p.add_argument("--seed", default="0")
    _add_rule_args(p)
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("validate", help="run the engine checks against exact answers")
    p.add_argument("--games", type=int, default=20_000)
    p.add_argument("--samples", type=int, default=200_000)
    p.add_argument("--quick", action="store_true", help="skip the slow 10x10 exact counts")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("race", help="two players take turns; who sinks the other first?")
    p.add_argument("a", help="player A's shooter")
    p.add_argument("b", help="player B's shooter")
    p.add_argument("--placer-a", default="uniform", help="how player A hides its fleet")
    p.add_argument("--placer-b", default="uniform", help="how player B hides its fleet")
    p.add_argument("-n", "--games", type=int, default=10_000)
    p.add_argument("--seed", default="0")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--salvo", action="store_true", help="one shot per ship afloat each turn")
    p.add_argument("--out", help="write the summary as JSON")
    _add_rule_args(p)
    p.set_defaults(func=cmd_race)

    p = sub.add_parser("series", help="many games against one opponent: does learning help?")
    p.add_argument("shooters", nargs="+", help="e.g. opponent-aware probability-map")
    p.add_argument("--placer", default="edges", help="the opponent's hiding style")
    p.add_argument("--series", type=int, default=1_000, help="how many opponents")
    p.add_argument("--length", type=int, default=20, help="games against each opponent")
    p.add_argument("--seed", default="0")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--out", help="write the summary as JSON")
    _add_rule_args(p)
    p.set_defaults(func=cmd_series)

    p = sub.add_parser("matrix", help="every shooter against every placer, and the equilibrium")
    p.add_argument("--shooters", required=True, help="comma-separated")
    p.add_argument("--placers", required=True, help="comma-separated")
    p.add_argument("-n", "--games", type=int, default=10_000, help="games per placer")
    p.add_argument("--seed", default="0")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--out", help="write the matrix and equilibrium as JSON")
    _add_rule_args(p)
    p.set_defaults(func=cmd_matrix)

    p = sub.add_parser("worst-case", help="search for a layout that is hard for one shooter")
    p.add_argument("shooter")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--games", type=int, default=50, help="shooter seeds per evaluation")
    p.add_argument("--seed", default="worst")
    p.add_argument("--out", help="write the result as JSON")
    _add_rule_args(p)
    p.set_defaults(func=cmd_worst_case)

    p = sub.add_parser("lab", help="run experiments continuously and publish live stats")
    p.add_argument("--config", default="experiments/live.toml")
    p.add_argument("--state-dir", default="lab-state")
    p.add_argument("--publish", default="lab-state/public/battleship-live.json")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--publish-every", type=float, default=60.0, help="seconds")
    p.add_argument("--slow-workers", type=int,
                   help="workers for lane = slow conditions (default: a quarter of --workers)")
    p.add_argument("--max-batches", type=int, help="stop after this many cycles (testing)")
    p.add_argument("--upload", help="rsync destination for the snapshot, e.g. host:/dir/")
    p.add_argument("--allow-dirty", action="store_true",
                   help="run even with uncommitted source changes (no commit is published)")
    p.set_defaults(func=cmd_lab)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except ValueError as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()
