# battleship-strategy-sim

Which strategy sinks a Battleship fleet in the fewest shots, and where should you hide
your own ships against it? This tool plays reproducible games of Battleship between
automated shooting and ship placement strategies and measures how many shots each one
needs.

It's the code behind the
[Battleship experiment](https://kalebhoffer.com/experiments/battleship-strategy) on
kalebhoffer.com. The full research plan is in [docs/research-plan.md](docs/research-plan.md).

> **Status: Phase 1 (engine and validation).** Only the two random baselines exist so
> far. No findings have been published.

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/kalebhoffer/battleship-strategy-sim
cd battleship-strategy-sim
uv sync

uv run battleship-sim list                     # shooters and placers
uv run battleship-sim validate                 # check the engine against exact answers
uv run battleship-sim show --seed 7            # print a random fleet
uv run battleship-sim run random random-repeat -n 100000 --seed 1
```

Each run writes a folder under `results/`:

| File | Contents |
| --- | --- |
| `games.csv` | One row per shooter per game: shots, first hit, repeats, and the shot each ship sank |
| `summary.json` | Shots per shooter (mean with 95% CI, median, percentiles, worst case), exact values where known, and paired differences between shooters |
| `run.json` | Everything needed to reproduce the run: shooters, placer, seed, rules, version, git commit |

## Fair comparisons

Every shooter in a run faces **the same fleets**. Game *i*'s fleet depends only on the
seed, the placer, and *i*, so adding a shooter never changes what the others face. The
differences between shooters are paired by game, which removes most of the luck of the
fleet. Results don't depend on how many CPUs ran them.

Shooters are built from the rules and a random number generator, and learn about the board
only through the answer to each shot. A test checks that strategy code cannot reach the
hidden layout.

## Rules

The baseline is the official Hasbro rule sheet. In the official game **every hit names
the ship**, not just the sinking shot, which is more information than most online analyses
assume. Variants are flags:

```bash
uv run battleship-sim run random --feedback sink-only    # ships named only when sunk
uv run battleship-sim run random --feedback hit-only     # hit or miss, nothing else
uv run battleship-sim run random --no-touching           # ships may not touch
uv run battleship-sim run random --size 8 --fleet 4,3,2  # other boards and fleets
```

Details, sources, and every simplification are in [docs/RULES.md](docs/RULES.md).

## Validation

`battleship-sim validate` runs the checks from section 5.3 of the research plan:

- Random fire matches the exact distribution of shots, not just its average.
- Random fire with repeats matches the coupon-collector formula.
- The number of legal fleet layouts is computed by two independent methods (a dynamic
  program and brute-force listing) on small boards, and on the full board it is checked
  against a Monte Carlo estimate.
- The uniform placer passes a chi-square test against every legal layout. The naive
  one-ship-at-a-time placer fails it, as it should.

```bash
uv run battleship-sim count                  # exact count of legal fleet layouts
uv run battleship-sim count --no-touching
```

## Development

```bash
uv run pytest -q
uv run ruff check .
```

## License

MIT
