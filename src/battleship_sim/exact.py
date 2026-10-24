"""Exact answers for the two random shooters, used to validate the simulation.

Random fire without repeats: the fleet is sunk on the shot that reaches the last of the
k ship cells in a random ordering of all n cells, so P(shots <= t) = C(t, k) / C(n, k)
and the mean is k(n + 1)/(k + 1). Ship positions don't matter, only how many cells they
cover.

Random fire with repeats: collecting k particular cells out of n with replacement takes
n/k + n/(k-1) + ... + n/1 = n * H(k) shots on average (a coupon-collector argument).
"""

from __future__ import annotations

from fractions import Fraction
from math import comb


def random_fire_cdf(ship_cells: int, cells: int) -> list[Fraction]:
    """cdf[t] = P(the fleet is sunk within t shots), for t = 0..cells."""
    total = comb(cells, ship_cells)
    return [Fraction(comb(t, ship_cells), total) for t in range(cells + 1)]


def random_fire_mean(ship_cells: int, cells: int) -> Fraction:
    return Fraction(ship_cells * (cells + 1), ship_cells + 1)


def random_fire_variance(ship_cells: int, cells: int) -> Fraction:
    k, n = ship_cells, cells
    return Fraction(k * (n - k) * (n + 1), (k + 1) ** 2 * (k + 2))


def random_repeat_mean(ship_cells: int, cells: int) -> Fraction:
    return sum((Fraction(cells, j) for j in range(1, ship_cells + 1)), Fraction(0))


def random_repeat_variance(ship_cells: int, cells: int) -> Fraction:
    # Sum of independent geometric waits with success chance j/n.
    return sum(
        (Fraction(1 - Fraction(j, cells)) / Fraction(j, cells) ** 2
         for j in range(1, ship_cells + 1)),
        Fraction(0),
    )
