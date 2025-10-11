from collections.abc import Sequence

from numba import njit
from numpy import empty, isnan
from numpy.typing import NDArray

from nested_mapping import NestedMapping


@njit
def weekly_to_daily(array: NDArray) -> NDArray:
    ret = empty(array.shape[0] * 7)

    for i in range(7):
        ret[i::7] = array

    return ret


@njit
def weeks_to_days(array: NDArray) -> NDArray:
    ret = empty(array.shape[0] * 7)

    for i in range(7):
        ret[i::7] = array
        ret[i::7] += i

    return ret


def refine_reactor_data2(
    source: NestedMapping,
    target: NestedMapping,
    *,
    reactors: Sequence[str],
    isotopes: Sequence[str],
    reactor_number_start: int = 1,
    clean_source: bool = True,
) -> None:
    for corename in reactors:
        week = source["week", corename]
        day = source["day", corename]
        core = source["core", corename]

        power = source["power", corename]
        fission_fractions = {key: source[key.lower()] for key in isotopes}

        ncores = 1
        for i in range(ncores):
            rweek = week[i::ncores]
            step = rweek[1:] - rweek[:-1]
            assert (step == 1).all(), "Expect reactor data for each week"

        target["days"] = (days_storage := {})

        key = (corename,)
        target[("power",) + key] = weekly_to_daily(power)
        for isotope in isotopes:
            target[("fission_fraction",) + key + (isotope,)] = weekly_to_daily(
                fission_fractions[isotope][corename]
            )

        days = weeks_to_days(day)
        days_stored = days_storage.setdefault(corename, days)
        if days is not days_stored:
            assert all(days == days_stored)

    for key, array in target.walkjoineditems():
        if isnan(array).any():
            raise ValueError(f"Invalid refined reactor data for {key}")

    if clean_source:
        for key in tuple(source.walkkeys()):
            source.delete_with_parents(key)
