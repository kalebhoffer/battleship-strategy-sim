# Battleship Strategy Research Plan

**Publication:** Kaleb Hoffer Research  
**Website:** [kalebhoffer.com](https://kalebhoffer.com)  
**Research type:** Computational experimentation, simulation, probability, search under uncertainty, and game theory  
**Status:** Research proposal. This document contains no findings.

## Central Research Question

> **Which strategy sinks a full Battleship fleet in the fewest shots, and where should you hide your own ships against it?**

Battleship is a small search problem. Every shot reveals a little information, and the shooter has to decide where to look next. The placer has the opposite job: arrange five ships so that a good searcher takes as long as possible to find them. Because the game is small, it can be simulated millions of times. It is still rich enough that strategy matters.

The research runs at three levels:

1. **Shooting:** How many shots does each search strategy need to sink a fleet, and how much better is each one than firing at random?
2. **Placement:** Can ship placement slow down a good shooter, and does a placement that works against one shooter fail against another?
3. **The full game:** Shooting and placement interact. What does the shooter-versus-placer matchup look like, and is there a stable best strategy for each side?

The goal is to play large numbers of reproducible Battleship games in Python, compare strategies statistically, and publish the results with the code, data, methodology, and limitations.

---

## 1. Baseline Rules

The baseline is the classic Hasbro game, checked against the Hasbro F4527 rule sheet (© 2022). The edition and each rule are recorded in `docs/RULES.md`.

| Rule | Baseline |
|---|---|
| Grid | 10×10, rows A–J, columns 1–10 |
| Fleet | Five ships of lengths 5, 4, 3, 3, 2 (17 cells). Ship names vary by edition. |
| Orientation | Horizontal or vertical. No diagonals. |
| Overlap | Not allowed |
| Touching | Allowed. Ships may sit next to each other. |
| Shots per turn | One |
| Feedback | Hit or miss on every shot. **A hit names the ship that was hit**, and the owner announces each ship that sinks. |
| Win | First player to sink all five enemy ships |

### 1.1 Rule Variants (flags, not baseline)

| Variant | Why test it |
|---|---|
| No touching | A common house rule. It changes the probability maps and makes some placements illegal. |
| Sink-only feedback | A hit is not named until the ship sinks. Most published analyses use this rule, so it is the main comparison point. |
| Hit-only feedback | The shooter learns only hit or miss and has to infer when a ship is finished. |
| Salvo | Printed on the rule sheet as an official variant. One shot per surviving ship each turn, which rewards sinking ships early. |
| Grid size | 8×8 and 12×12, to see whether the ranking of strategies depends on board density |
| Fleet composition | Smaller and larger fleets on the same grid |

---

## 2. Shooting Strategies

Every shooter has the same interface: it sees the shot history and its feedback, and it returns the next cell. No shooter may look at the hidden board.

| Strategy | Behavior | Research question |
|---|---|---|
| Random with repeats | Picks any cell, even one already fired at | How bad is truly naive play? Used as a floor. |
| Random | Picks a random cell it has not fired at | The real baseline. Every other shooter is compared to this one. |
| Hunt and target | Fires at random until it hits, then works through the neighbors until the ship is sunk | How much does following up on a hit gain? |
| Hunt and target + parity | Hunts only on a checkerboard pattern, since every ship covers at least two cells | How much does parity add? |
| Adaptive parity | Spaces the hunt pattern to the smallest ship still afloat | Is it worth widening the pattern as small ships sink? |
| Human patterns | Fixed sweeps: diagonals, spirals, edges first, center first | Do common human habits beat random hunting? |
| Probability map | Counts every placement of each remaining ship that fits the evidence and fires at the cell covered most often | The standard strong strategy. How much better is it than parity? |
| Full Bayesian | Samples whole fleet layouts consistent with the evidence, so ships constrain each other, and fires at the cell most likely to be occupied | Does modeling the whole fleet beat counting ship by ship? |
| Information gain | Fires where the expected reduction in uncertainty (entropy) is largest, not where a hit is most likely | Is aiming for information better than aiming for hits? |
| Opponent-aware | Like the probability map, but weighted by placements seen in earlier games against the same opponent | Can a shooter learn a placer's habits over a series? |

The probability-map and Bayesian shooters must use exactly the information the rules allow. When a variant hides sink announcements, the shooter cannot use them.

---

## 3. Placement Strategies

| Strategy | Behavior | Research question |
|---|---|---|
| Uniform random | Picks uniformly from all legal fleet layouts | The neutral baseline |
| Naive random | Places each ship at a random legal spot, one after another | Placing ships one by one is not uniform over whole layouts. Does the bias matter? |
| Edges and corners | Prefers cells along the border | Does hiding on the edges help, and against which shooters? |
| Center | Prefers the middle of the board | A control for edge-hugging |
| Clustered | Puts ships next to each other | Does touching confuse hunt-and-target follow-ups? |
| Spread out | Keeps maximum distance between ships | Does spreading out slow the hunt phase? |
| Anti-density | Favors cells that a probability-map shooter will fire at last | Can a placer exploit a known strong shooter? |
| Adversarial | Searches for layouts that make a specific shooter take the most shots | What is the worst case for each shooter? |
| Mixed / equilibrium | Randomizes over placement styles in proportions that come out of the game-theory study | What does a placement look like if it can't be exploited? |

---

## 4. Proposed Publication Series

Each study is a candidate post on kalebhoffer.com. The first three cover the experiment page's current hypothesis.

| # | Working title | Core question |
|---|---|---|
| 1 | How Bad Is Random Fire? | The baseline distribution, checked against the exact math |
| 2 | Hunt, Target, and the Checkerboard | How much follow-up and parity save over random fire |
| 3 | Probability Maps | Whether counting placements beats parity, and by how much |
| 4 | Where Should You Hide Your Ships? | Each placement style against each shooter |
| 5 | The Shooter–Placer Game | The full payoff matrix and whether either side has a stable best strategy |
| 6 | Does Going First Matter? | Head-to-head races and first-move advantage |
| 7 | House Rules Change the Answer | Whether the rankings hold under the variants in 1.1 |
| 8 | Learning Your Opponent | Whether an opponent-aware shooter beats a placer with habits over a series of games |

---

## 5. Python Simulation Architecture

Same conventions as the Monopoly simulator: Python 3.12, uv, src layout, MIT license, pytest + ruff, GitHub Actions CI, and no runtime dependencies if pure Python is fast enough.

```
battleship_sim/
├── board.py        # grid, ships, legal placements (bitmask representation)
├── rules.py        # Rules dataclass with variant flags
├── game.py         # single-shooter solve loop and two-player race
├── shooters/       # one module per shooting strategy
├── placers/        # one module per placement strategy
├── stats.py        # paired comparisons, confidence intervals
├── lab.py          # long-running live lab (same shape as monopoly-sim lab)
└── cli.py          # battleship-sim
```

### 5.1 Two Kinds of Experiment

1. **Solve:** One shooter against one hidden fleet, counting shots until all ships are sunk. Most of the studies use this.
2. **Race:** Two players, each with a shooter and a placer, take turns until one fleet is sunk. Studies 6 and 7 (Salvo) need this.

### 5.2 Fair Comparisons

- **Common fleets:** In a shooting comparison, every shooter faces the same set of fleets, generated from one seed. Comparisons are paired, which removes most fleet-to-fleet noise.
- **Seeds everywhere:** Every run records the seeds, the engine version, and the full config, and it can be reproduced exactly.
- **No peeking:** Shooters receive only an observation object (shots and feedback). A test asserts that no shooter can reach the hidden board.

### 5.3 Validation Before Any Results

- Random fire without repeats has a known expected shot count: the position of the last of 17 ship cells in a random ordering of 100 cells. The simulation must match the formula.
- Count the legal layouts for each ship and for the whole fleet, and check the counts against an independent calculation.
- Show that the uniform placer is uniform and that the naive placer is not, using a chi-square test on cell occupancy.
- Run hand-built boards with known optimal shot sequences as regression tests.

### 5.4 Performance

The full Bayesian shooter is the expensive one. Start with rejection sampling and switch to MCMC only if profiling shows it is needed. Report how many samples each decision uses, and test whether the result is sensitive to that number.

---

## 6. Simulation Scale

| Experiment | Games per condition (starting target) |
|---|---|
| Cheap shooters (random, hunt/target, parity) | 1,000,000 |
| Probability map | 100,000–1,000,000 |
| Full Bayesian / information gain | 10,000–100,000, depending on speed |
| Placement × shooter matrix | 100,000 per cell |
| Head-to-head races | 1,000,000 per pairing |

Stop adding games once the confidence interval on the mean shot count is narrower than 0.1 shots.

---

## 7. Outcome Metrics

### Primary Metric

**Shots to sink the whole fleet** (17 to 100).

### Secondary Metrics

- Full distribution: mean, median, standard deviation, 5th and 95th percentiles, and worst case seen
- Shots to the first hit, and shots to sink each ship
- Hit rate during hunting and during targeting
- Head-to-head win rate and first-mover advantage
- Decision time per shot, which matters for the expensive shooters

### Statistical Reporting

- 95% confidence intervals on every mean
- Paired differences between shooters on common fleets, with confidence intervals
- Win rates with Wilson intervals
- Unfinished games are not possible in Battleship, since every shooter finishes within 100 shots without repeats. Random-with-repeats is capped and reported separately.

---

## 8. Main Research Hypothesis

From the experiment page:

> I expect a probability-map strategy to sink a full fleet in noticeably fewer shots than hunt-and-target, and both to beat random fire by a wide margin.

Secondary hypotheses for the placement studies:

- Against random and hunt-and-target shooters, placement style hardly matters.
- Against a probability-map shooter, edge and anti-density placements help, but a shooter that knows about the bias can exploit them.
- The shooter–placer game has no single best pure placement. The stable answer is a mixed strategy.

---

## 9. Implementation Order

### Phase 1: Build and Validate the Engine

Board, rules, solve loop, random and uniform placers, the random shooters, and every validation check in 5.3. CLI and CI.

### Phase 2: Studies 1–3

Hunt and target, parity, adaptive parity, human patterns, and the probability map. Paired runs on common fleets. These cover the page's hypothesis.

### Phase 3: Placement (Studies 4–5)

All placers, the full shooter × placer matrix, and the adversarial placer. Then solve for the shooter–placer equilibrium on the payoff matrix.

### Phase 4: Races, Variants, and Learning (Studies 6–8)

The two-player race loop, rule variant flags, the full Bayesian and information-gain shooters, and the opponent-aware shooter.

### Live Lab

Deploy a `battleship-sim lab` container on the VPS after Phase 2, the same way as the Monopoly lab. It writes `battleship-live.json`, and the experiment page shows it as running or preliminary until the analysis is written.

---

## 10. Publication Deliverables

For each study:

- The experiment page sections (question, hypothesis, methodology, results, interpretation, limitations, conclusion, reproduce)
- Code at a tagged commit, with the exact command and seeds
- Raw results (CSV or JSON) small enough to host, plus a script that regenerates them
- Charts: shot-count distributions, cumulative "fleet sunk by shot N" curves, and per-cell heat maps of shots and of placements

---

## 11. References and Starting Points

- Hasbro Battleship rulebook (edition to be recorded in `docs/RULES.md`)
- Nick Berry, "Battleship," DataGenetics (2011): the well-known hunt/target, parity, and probability-density comparison. Treat it as prior work to reproduce independently, not as a source of numbers.
- Standard references on Bayesian search theory and on optimal search under uncertainty
