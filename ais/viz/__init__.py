from __future__ import annotations

from typing import TypeAlias, Tuple

from numpy import ndarray, asarray

Color: TypeAlias = str | float | Tuple[float, float, float] | Tuple[float, float, float, float]


def vec(first: float, *others: float) -> ndarray:
    """ handy variadic constructor for numpy arrays"""
    return asarray((first,) + others)
