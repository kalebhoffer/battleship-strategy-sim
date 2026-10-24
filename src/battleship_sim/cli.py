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
        print(f"  {spec.name:<16}{spec.description}")
    print("\nplacers")
    for spec in PLACERS.values():
        print(f"  {spec.name:<16}{spec.description}")


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

    p = sub.add_parser("lab", help="run experiments continuously and publish live stats")
    p.add_argument("--config", default="experiments/live.toml")
    p.add_argument("--state-dir", default="lab-state")
    p.add_argument("--publish", default="lab-state/public/battleship-live.json")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--publish-every", type=float, default=60.0, help="seconds")
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
