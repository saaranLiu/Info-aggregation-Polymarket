"""Presentation precision for the manuscript; saved estimates remain unchanged."""
import math


def decimal(value, signed=False):
    value = float(value)
    if not math.isfinite(value):
        return "---"
    # Avoid a negative zero after rounding.
    if abs(value) < 0.05:
        value = 0.0
    return format(value, "+.1f" if signed else ".1f")


def tex_decimal(value, signed=False):
    return decimal(value, signed).replace("-", r"\ensuremath{-}")


def probability(value):
    """One decimal in scientific notation preserves small, nonzero p/q values."""
    value = float(value)
    if 0 < value < 0.1:
        mantissa, exponent = f"{value:.1e}".split("e")
        return rf"{mantissa}\times10^{{{int(exponent)}}}"
    return decimal(value)


def cell(estimate, se):
    return f"{tex_decimal(estimate)} ({tex_decimal(se)})"
