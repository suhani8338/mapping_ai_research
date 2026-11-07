"""Statistical helpers for count data."""

from __future__ import annotations

from collections.abc import Sequence


def chi_square_2x2(table: Sequence[Sequence[int]]) -> tuple[float, float]:
    """Return Pearson's chi-square statistic and p-value for a 2x2 table."""
    if len(table) != 2 or any(len(row) != 2 for row in table):
        raise ValueError("table must be 2x2")
    if any(value < 0 for row in table for value in row):
        raise ValueError("cell counts must be non-negative")
    total = sum(sum(row) for row in table)
    if total == 0:
        raise ValueError("table must contain at least one observation")
    row_totals = [sum(row) for row in table]
    col_totals = [table[0][column] + table[1][column] for column in range(2)]
    if 0 in row_totals or 0 in col_totals:
        return 0.0, 1.0
    statistic = 0.0
    for row in range(2):
        for column in range(2):
            expected = row_totals[row] * col_totals[column] / total
            observed = table[row][column]
            statistic += (observed - expected) ** 2 / expected
    # For one degree of freedom, the chi-square survival function has this form.
    p_value = 1.0 - _erf_approx((statistic / 2) ** 0.5)
    return statistic, max(0.0, min(1.0, p_value))


def _erf_approx(value: float) -> float:
    """Numerical approximation to erf, sufficient for the 1-df chi-square tail."""
    sign = 1 if value >= 0 else -1
    x = abs(value)
    t = 1.0 / (1.0 + 0.3275911 * x)
    polynomial = (
        (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t)
        - 0.284496736)
        * t
        + 0.254829592
    ) * t
    return sign * (1.0 - polynomial * __import__("math").exp(-(x * x)))


def fisher_exact_2x2(table: Sequence[Sequence[int]]) -> tuple[float, float]:
    """Return odds ratio and two-sided Fisher exact p-value for a 2x2 table."""
    if len(table) != 2 or any(len(row) != 2 for row in table):
        raise ValueError("table must be 2x2")
    if any(not isinstance(value, int) or value < 0 for row in table for value in row):
        raise ValueError("cell counts must be non-negative integers")
    a, b = table[0]
    c, d = table[1]
    odds_ratio = (a * d / (b * c)) if b * c else float("inf") if a * d else 0.0
    row_one = a + b
    row_two = c + d
    col_one = a + c
    total = row_one + row_two
    if total == 0:
        raise ValueError("table must contain at least one observation")
    lower = max(0, col_one - row_two)
    upper = min(row_one, col_one)
    observed_probability = _hypergeometric_probability(a, row_one, col_one, total)
    p_value = sum(
        _hypergeometric_probability(x, row_one, col_one, total)
        for x in range(lower, upper + 1)
        if _hypergeometric_probability(x, row_one, col_one, total)
        <= observed_probability + 1e-12
    )
    return odds_ratio, min(1.0, p_value)


def _hypergeometric_probability(x: int, row_one: int, col_one: int, total: int) -> float:
    from math import comb

    return comb(col_one, x) * comb(total - col_one, row_one - x) / comb(total, row_one)
