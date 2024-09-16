from collections.abc import Sequence

from numba import njit
from numpy import empty
from numpy.typing import NDArray

from multikeydict.nestedmkdict import NestedMKDict


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


def refine_reactor_data(
    source: NestedMKDict,
    target: NestedMKDict,
    *,
    reactors: Sequence[str],
    isotopes: Sequence[str],
    periods: Sequence[int] = (6, 8, 7),
    reactor_number_start: int = 1,
    clean_source: bool = True,
) -> None:
    week = source["week"]
    day = source["day"]
    core = source["core"]
    ndet = source["ndet"]

    power = source["power"]
    fission_fractions = {key: source[key.lower()] for key in isotopes}

    ncores = 6
    for i in range(ncores):
        rweek = week[i::ncores]
        step = rweek[1:] - rweek[:-1]
        assert (step == 1).all(), "Expect reactor data for each week"

    target["days"] = (days_storage := {})
    for period in periods:
        mask_period = ndet == period
        periodname = f"{period}AD"

        for icore, corename in enumerate(reactors, reactor_number_start):
            mask = mask_period * (core == icore)
            key = (
                periodname,
                corename,
            )
            target[("power",) + key] = weekly_to_daily(power[mask])
            for isotope in isotopes:
                target[("fission_fraction",) + key + (isotope,)] = weekly_to_daily(
                    fission_fractions[isotope][mask]
                )

            days = weeks_to_days(day[mask])
            days_stored = days_storage.setdefault(periodname, days)
            if days is not days_stored:
                assert all(days == days_stored)

    if clean_source:
        for key in tuple(source.walkkeys()):
            source.delete_with_parents(key)


def refine_reactor_data2(
    source: NestedMKDict,
    target: NestedMKDict,
    *,
    reactors: Sequence[str],
    isotopes: Sequence[str],
    reactor_number_start: int = 1,
    clean_source: bool = True,
) -> None:
    week = source["week"]
    day = source["day"]
    core = source["core"]

    power = source["power"]
    fission_fractions = {key: source[key.lower()] for key in isotopes}

    ncores = 1
    for i in range(ncores):
        rweek = week[i::ncores]
        step = rweek[1:] - rweek[:-1]
        assert (step == 1).all(), "Expect reactor data for each week"

    target["days"] = (days_storage := {})

    for icore, corename in enumerate(reactors, reactor_number_start):
        key = (corename,)
        target[("power",) + key] = weekly_to_daily(power)
        for isotope in isotopes:
            target[("fission_fraction",) + key + (isotope,)] = weekly_to_daily(
                fission_fractions[isotope]
            )

        days = weeks_to_days(day)
        days_stored = days_storage.setdefault(corename, days)
        if days is not days_stored:
            assert all(days == days_stored)

    if clean_source:
        for key in tuple(source.walkkeys()):
            source.delete_with_parents(key)
