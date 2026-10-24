"""Two-player games: both sides hide a fleet and take turns shooting (Studies 6 and 7).

The first player to sink the other's whole fleet wins, so a game can end before the
second player gets an equal number of turns. That is the first-move advantage Study 6
measures. Seats alternate: in even-numbered games player A goes first, in odd ones B,
so over an even number of games each goes first equally often.

Salvo (an official variant on the rule sheet): each turn you fire one shot per ship you
still have afloat, one at a time, hearing the result after each.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

from battleship_sim.board import Layout
from battleship_sim.game import Ocean, Shooter
from battleship_sim.placers import get_placer
from battleship_sim.rules import Rules
from battleship_sim.shooters import make_shooter
from battleship_sim.stats import wilson_interval

MAX_TURNS = 10_000  # only reachable by shooters that may repeat cells


@dataclass(frozen=True, slots=True)
class RaceResult:
    winner: int  # 0 = player A, 1 = player B
    first: int  # who fired first
    turns: int  # turns taken, both players together
    shots: tuple[int, int]  # shots each player fired
    afloat: tuple[int, int]  # ships each player still had at the end
    capped: bool = False


def race(
    layouts: tuple[Layout, Layout],
    shooters: tuple[Shooter, Shooter],
    rules: Rules,
    first: int = 0,
    max_turns: int = MAX_TURNS,
) -> RaceResult:
    """Player i hides layouts[i] and shoots with shooters[i] at the other player's fleet."""
    oceans = (Ocean(layouts[0], rules), Ocean(layouts[1], rules))
    shots = [0, 0]
    player = first
    turns = 0
    while turns < max_turns:
        turns += 1
        target = oceans[1 - player]
        volley = oceans[player].afloat if rules.salvo else 1
        for _ in range(volley):
            cell = shooters[player].fire()
            shots[player] += 1
            told, _ = target.fire(cell)
            shooters[player].observe(cell, told)
            if not target.afloat:
                return RaceResult(player, first, turns, (shots[0], shots[1]),
                                  (oceans[0].afloat, oceans[1].afloat))
        player = 1 - player
    winner = 0 if oceans[1].afloat < oceans[0].afloat else 1
    return RaceResult(winner, first, turns, (shots[0], shots[1]),
                      (oceans[0].afloat, oceans[1].afloat), capped=True)


@dataclass(frozen=True)
class RaceConfig:
    shooters: tuple[str, str]  # player A, player B
    placers: tuple[str, str] = ("uniform", "uniform")
    games: int = 10_000
    seed: int | str = 0
    rules: Rules = field(default_factory=Rules)
    workers: int = 0
    chunk_size: int = 1_000


def play_race(config: RaceConfig, game_index: int) -> RaceResult:
    layouts = tuple(
        get_placer(config.placers[side])(
            config.rules, random.Random(f"{config.seed}:fleet:{side}:{config.placers[side]}:"
                                        f"{game_index}")
        )
        for side in (0, 1)
    )
    shooters = tuple(
        make_shooter(config.shooters[side], config.rules,
                     random.Random(f"{config.seed}:shooter:{side}:{config.shooters[side]}:"
                                   f"{game_index}"))
        for side in (0, 1)
    )
    return race(layouts, shooters, config.rules, first=game_index % 2)


def _chunk(config: RaceConfig, start: int, stop: int) -> list[tuple[int, RaceResult]]:
    return [(i, play_race(config, i)) for i in range(start, stop)]


def run_races(config: RaceConfig) -> Iterator[tuple[int, RaceResult]]:
    for side in (0, 1):
        make_shooter(config.shooters[side], config.rules, random.Random(0))
        get_placer(config.placers[side])
    chunks = [(s, min(s + config.chunk_size, config.games))
              for s in range(0, config.games, config.chunk_size)]
    workers = config.workers or os.cpu_count() or 1
    if workers == 1:
        for start, stop in chunks:
            yield from _chunk(config, start, stop)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in [pool.submit(_chunk, config, a, b) for a, b in chunks]:
            yield from future.result()


@dataclass
class RaceTally:
    """Running totals, from which every published race statistic is computed."""

    games: int = 0
    wins: list[int] = field(default_factory=lambda: [0, 0])
    first_wins: int = 0  # games won by whoever fired first
    went_first: list[int] = field(default_factory=lambda: [0, 0])
    wins_going_first: list[int] = field(default_factory=lambda: [0, 0])
    turns: int = 0  # total, for the mean
    capped: int = 0

    def add(self, r: RaceResult) -> None:
        self.games += 1
        self.wins[r.winner] += 1
        self.went_first[r.first] += 1
        if r.winner == r.first:
            self.first_wins += 1
            self.wins_going_first[r.first] += 1
        self.turns += r.turns
        self.capped += r.capped

    def summary(self, names: tuple[str, str]) -> dict:
        n = self.games
        players = []
        for side in (0, 1):
            low, high = wilson_interval(self.wins[side], n)
            first_n = self.went_first[side]
            second_n = n - first_n
            players.append({
                "name": names[side],
                "wins": self.wins[side],
                "win_rate": self.wins[side] / n if n else None,
                "ci95": [low, high],
                "win_rate_going_first": (self.wins_going_first[side] / first_n
                                         if first_n else None),
                "win_rate_going_second": ((self.wins[side] - self.wins_going_first[side])
                                          / second_n if second_n else None),
            })
        low, high = wilson_interval(self.first_wins, n)
        mean_turns = self.turns / n if n else None
        return {
            "games": n,
            "players": players,
            "first_player_win_rate": self.first_wins / n if n else None,
            "first_player_ci95": [low, high],
            "mean_turns": mean_turns,
            "capped": self.capped,
        }
