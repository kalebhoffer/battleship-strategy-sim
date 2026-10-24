# battleship-strategy-sim

Which strategy sinks a Battleship fleet in the fewest shots, and where should you hide
your own ships against it? This tool plays reproducible games of Battleship between
automated shooting and ship placement strategies and measures how many shots each one
needs.

It's the code behind the
[Battleship experiment](https://kalebhoffer.com/experiments/battleship-strategy) on
kalebhoffer.com. The full research plan is in [docs/research-plan.md](docs/research-plan.md).

> **Status: all four phases built (engine, validation, shooters, placement styles,
> the shooter-placer game, two-player races, Salvo, Bayesian shooters, and learning
> across games) plus a live lab.** No findings have been published.

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

## Shooters

| Name | Strategy |
| --- | --- |
| `random-repeat` | Any cell at random, even ones already fired at. The floor. |
| `random` | A random cell not fired at yet. The baseline. |
| `hunt-target` | Random search. After a hit, fires at the neighbors (up, right, down, left) until none are left. |
| `hunt-target-parity` | Hunt and target, searching one color of a checkerboard. Every ship covers both colors. |
| `adaptive-parity` | Hunt and target on diagonal stripes spaced by the shortest ship still afloat. |
| `pattern-rows` | Hunt and target, searching A1, A2, ... row by row. |
| `pattern-diagonal` | Hunt and target, sweeping every other diagonal from the top left. |
| `pattern-spiral-in` | Hunt and target, spiraling from the edges to the center. |
| `pattern-spiral-out` | Hunt and target, spiraling from the center to the edges. |
| `probability-map` | Counts every position of every ship that fits what it has seen, and fires at the cell most likely to hold a ship. |
| `bayes` | Samples whole fleets consistent with everything it has been told (Gibbs sampling, checked against exact posteriors) and fires at the likeliest square. `bayes:N` sets the sampler sweeps per shot. |
| `info-gain` | Same belief as `bayes`, but fires where the answer is most uncertain (highest outcome entropy). |
| `opponent-aware` | A probability map that learns where one opponent hides ships over a series of games. |

Shooters with a hunt mode and a target mode also report hit rates for each mode.
Strategy code is in [`shooters/`](src/battleship_sim/shooters).

## Placers

| Name | How the fleet is hidden |
| --- | --- |
| `uniform` | Every legal layout equally likely. The neutral baseline. |
| `naive` | One ship at a time, each at a random spot that still fits. Not uniform. |
| `edges` | Every ship touches the outer edge. Uniform among such layouts. |
| `center` | Every ship inside the middle 6x6. Uniform among such layouts. Impossible without touching (there are 0 such layouts). |
| `clustered` | One group, each ship touching another side to side. |
| `spread` | The most spread out of 50 random layouts. |
| `anti-density` | Each ship in the quarter of its positions the probability map rates least likely before the first shot. |
| `mix:edges=0.3,uniform=0.7` | A random style each game, with those weights |

## The shooter-placer game

```bash
# Study 4/5: every shooter against every placer, plus the equilibrium of the zero-sum game
uv run battleship-sim matrix --shooters random,hunt-target,hunt-target-parity,probability-map \
  --placers uniform,naive,edges,center,clustered,spread,anti-density -n 100000 --seed study4 \
  --out results/matrix.json

# A layout that is hard for one shooter (hill-climbing search, re-checked on fresh games)
uv run battleship-sim worst-case probability-map --steps 500 --games 100
```

The equilibrium solver reports the mixes and a lower and upper bound on the game's value,
so you can see how close the approximation is. The `mix:` placer can then play the
equilibrium mix.

## Two players, Salvo, and learning (Studies 6-8)

```bash
# Study 6: does going first matter? Each player goes first in half the games.
uv run battleship-sim race probability-map probability-map -n 100000 --seed study6

# Study 7: Salvo, the official variant with one shot per ship still afloat
uv run battleship-sim race probability-map hunt-target-parity -n 100000 --seed study7 --salvo

# Study 8: 20 games each against 1,000 opponents who always hide ships on the edges
uv run battleship-sim series opponent-aware probability-map --placer edges --series 1000
```

The race engine is checked against an exact answer: two random shooters, where the first
player wins exactly when it needs no more shots than the second.

## Studies 1-3

Each study is one run, so every shooter faces the same fleets. A million games of the
Study 2 lineup takes about 11 minutes on a 10-core laptop and writes a ~500 MB CSV.

```bash
# Study 1: how bad is random fire?
uv run battleship-sim run random-repeat random -n 1000000 --seed study1

# Study 2: hunt, target, and the checkerboard (plus human search patterns)
uv run battleship-sim run random hunt-target hunt-target-parity adaptive-parity \
  pattern-rows pattern-diagonal pattern-spiral-in pattern-spiral-out -n 1000000 --seed study2

# Study 3: probability maps, under the official rules and the common sink-only rule
uv run battleship-sim run hunt-target hunt-target-parity adaptive-parity probability-map \
  -n 1000000 --seed study3
uv run battleship-sim run hunt-target hunt-target-parity adaptive-parity probability-map \
  -n 1000000 --seed study3 --feedback sink-only
```

## Live lab

`battleship-sim lab` plays the conditions in [`experiments/live.toml`](experiments/live.toml)
forever, in batches, and writes a JSON snapshot every minute: means with confidence
intervals, medians, percentiles, the distribution of shots, paired differences, and the
exact command that reproduces every number. The kalebhoffer.com experiment page shows it,
labeled as preliminary.

```bash
uv run battleship-sim lab --workers 8                       # snapshot in lab-state/public/
uv run battleship-sim lab --workers 8 --upload host:/dir/   # also rsync it to a web server
```

- **Reproducible.** Each condition has its own seed and numbers its games from 0, so the
  snapshot's `reproduce` command replays exactly the games behind it. A test checks that
  the lab's statistics match an offline `run` of the same games.
- **Never mixes engine versions.** Each condition is fingerprinted by replaying fixed
  canary games. If a code change alters how a condition plays, its totals are archived
  and it starts over. Conditions the change didn't affect keep counting.
- **Refuses uncommitted code**, so every published number traces to a commit.

[`deploy/macos/`](deploy/macos) runs it as a low-priority background job on a Mac.

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
