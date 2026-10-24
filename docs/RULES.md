# Rules as implemented

The baseline follows the **Hasbro Battleship F4527 rule sheet (en-us, © 2022)**, downloaded
from Hasbro consumer care on 2026-10-24. The older A3264 sheet (© 2018) was checked too
and says the same thing on every point below. The PDFs are not in the repository because
of Hasbro's copyright. Their SHA-256 hashes, for reference:

| File | SHA-256 |
| --- | --- |
| `F4527_en-us.pdf` | `9618620f0e9f0924a3122272fc3e4fd2c40651ddf9cd45ae15031d6440db22d6` |
| `A3264_en-us.pdf` (Spanish/Portuguese text despite the name) | `06437b5841a4bdb22c346f2ae3ea651f15ea3f10083362c4e7000738101326ca` |

Put local copies in `docs/_rulebook/`, which is gitignored.

## Baseline

| Rule | Rule sheet | Engine |
| --- | --- | --- |
| Grid | 10×10, rows A–J, columns 1–10 | `Rules.size = 10` |
| Fleet | One each of Carrier (5), Battleship (4), Destroyer (3), Submarine (3), Patrol Boat (2) | `rules.FLEET` |
| Orientation | "horizontally or vertically, but not diagonally" | `board.placements` |
| Off the grid | No part of a ship may overlap the letters or numbers | Placements stay in bounds |
| Overlap | Not stated. Physically impossible, since each ship's pegs go in their own holes. | Not allowed |
| Touching | Not prohibited | Allowed (`touching=True`) |
| Shots | One shot per turn | One shot per `fire()` |
| On a hit | "Your opponent tells you which ship you hit" | `feedback="named-hits"`: every hit reports the ship |
| On a sink | "The ship's owner announces which ship was sunk" | The sinking shot reports the sunk ship |
| Win | Sink all five enemy ships first | The game ends when the last ship sinks |

**Important:** in the official game every hit names the ship. Many online analyses of
Battleship assume the shooter learns only "hit" until a ship sinks. That is a variant here
(`sink-only`), not the baseline. Results under the two rules are not directly comparable.

## Variants

| Flag | Meaning | Source |
| --- | --- | --- |
| `--feedback sink-only` | Hit or miss. A ship is named only when it sinks. | Common in published analyses |
| `--feedback hit-only` | Hit or miss, and nothing else, not even sinks | Stricter house rule |
| `--no-touching` | Ships may not touch, including diagonally | Common house rule |
| `--size N`, `--fleet 5,4,3,3,2` | Other grids and fleets | For the robustness study |

Salvo is printed on the rule sheet as an official variant: one shot per surviving ship
each turn, starting at five, and "for a more challenging Salvo game, don't announce which
ships are hit." It needs the two-player race loop and arrives in Phase 4.

## Simplifications and choices

- **Repeated shots.** The rule sheet assumes players never repeat a shot, because the pegs
  record them. Only the `random-repeat` floor repeats. A repeated shot on a ship is reported
  as a hit again but does no extra damage.
- **Shot cap.** Shooters that can repeat stop at 10,000 shots. These games are flagged
  `capped` and reported separately. A shooter that never repeats always finishes within
  100 shots.
- **Layout counts treat ships as labeled.** Swapping the Destroyer and the Submarine gives
  a different layout, because the rules name the ship on every hit.
- **Uniform placement** means every legal layout of the whole fleet is equally likely. It
  uses rejection sampling, which is exact. Placing ships one at a time (`naive`) is not
  uniform, and the validation suite shows the difference.
