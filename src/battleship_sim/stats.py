"""Small statistics helpers, written out so the package has no dependencies."""

from __future__ import annotations

import math
from collections.abc import Sequence

Z95 = 1.959963984540054


def mean_ci(values: Sequence[float], z: float = Z95) -> tuple[float, float, float]:
    """Mean and a normal-approximation 95% confidence interval."""
    n = len(values)
    if n == 0:
        return (math.nan, math.nan, math.nan)
    mean = sum(values) / n
    if n == 1:
        return (mean, mean, mean)
    half = z * stdev(values, mean) / math.sqrt(n)
    return (mean, mean - half, mean + half)


def stdev(values: Sequence[float], mean: float | None = None) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    if mean is None:
        mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile of already-sorted values."""
    if not sorted_values:
        return math.nan
    pos = q * (len(sorted_values) - 1)
    low = math.floor(pos)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (pos - low)


def wilson_interval(wins: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95% confidence interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def chi_square_p(observed: Sequence[int], expected: Sequence[float]) -> tuple[float, float]:
    """Pearson chi-square statistic and its p-value (degrees of freedom = bins - 1)."""
    stat = sum((o - e) ** 2 / e for o, e in zip(observed, expected, strict=True))
    return stat, gamma_q((len(observed) - 1) / 2, stat / 2)


def gamma_q(a: float, x: float) -> float:
    """Regularized upper incomplete gamma function Q(a, x)."""
    if x <= 0:
        return 1.0
    log_front = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1:  # series for P(a, x)
        term = total = 1 / a
        k = a
        for _ in range(10_000):
            k += 1
            term *= x / k
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return 1 - total * math.exp(log_front)
    # Lentz's continued fraction for Q(a, x)
    tiny = 1e-300
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return h * math.exp(log_front)


def ks_statistic(sorted_values: Sequence[int], cdf: Sequence[float]) -> float:
    """Largest gap between the empirical CDF of integer samples and an exact CDF.

    cdf[t] = P(X <= t). For a discrete distribution the usual 1.36/sqrt(n) cutoff is
    conservative.
    """
    n = len(sorted_values)
    worst = 0.0
    i = 0
    for t, p in enumerate(cdf):
        while i < n and sorted_values[i] <= t:
            i += 1
        worst = max(worst, abs(i / n - float(p)))
    return worst
