"""Game rules, rule variants, and the feedback a shooter is allowed to see.

The baseline follows the Hasbro rule sheet listed in EDITION. See docs/RULES.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

EDITION = "Hasbro Battleship F4527 rule sheet (en-us, (c) 2022)"


@dataclass(frozen=True, slots=True)
class Ship:
    name: str
    length: int


FLEET = (
    Ship("Carrier", 5),
    Ship("Battleship", 4),
    Ship("Destroyer", 3),
    Ship("Submarine", 3),
    Ship("Patrol Boat", 2),
)


class Feedback(StrEnum):
    """How much the defender tells the shooter after each shot."""

    NAMED_HITS = "named-hits"  # rulebook: every hit names the ship, sinks are announced
    SINK_ONLY = "sink-only"  # hit or miss; a ship is named only when it sinks
    HIT_ONLY = "hit-only"  # hit or miss and nothing else


@dataclass(frozen=True, slots=True)
class Rules:
    size: int = 10
    fleet: tuple[Ship, ...] = FLEET
    feedback: Feedback = Feedback.NAMED_HITS
    touching: bool = True  # False = ships may not touch, even diagonally
    # Two-player games only: one shot per ship you still have afloat, each turn.
    salvo: bool = False

    def __post_init__(self) -> None:
        if not 2 <= self.size <= 26:
            raise ValueError("grid size must be between 2 and 26")
        if not self.fleet:
            raise ValueError("fleet is empty")
        if len({s.name for s in self.fleet}) != len(self.fleet):
            raise ValueError("ship names must be unique")
        for ship in self.fleet:
            if not 1 <= ship.length <= self.size:
                raise ValueError(f"{ship.name} does not fit on a {self.size}x{self.size} grid")
        if self.fleet_cells > self.size * self.size:
            raise ValueError("fleet has more cells than the grid")

    @property
    def cells(self) -> int:
        return self.size * self.size

    @property
    def fleet_cells(self) -> int:
        return sum(s.length for s in self.fleet)

    def describe(self) -> dict[str, object]:
        return {
            "edition": EDITION,
            "size": self.size,
            "fleet": [{"name": s.name, "length": s.length} for s in self.fleet],
            "feedback": str(self.feedback),
            "touching": self.touching,
            # Listed only when on, so conditions defined before Salvo existed keep their
            # fingerprints (lab.py hashes this description).
            **({"salvo": True} if self.salvo else {}),
        }


def parse_fleet(spec: str) -> tuple[Ship, ...]:
    """'5,4,3,3,2' -> the official fleet; other lengths get generic names."""
    lengths = tuple(int(x) for x in spec.split(",") if x.strip())
    if lengths == tuple(s.length for s in FLEET):
        return FLEET
    return tuple(Ship(f"Ship {i + 1} ({n})", n) for i, n in enumerate(lengths))


@dataclass(frozen=True, slots=True)
class ShotResult:
    """What the shooter is told. Fields the rules hide are None."""

    hit: bool
    ship: int | None = None  # fleet index of the ship hit, if named
    sunk: int | None = None  # fleet index of the ship sunk by this shot, if announced
