from collections.abc import Sequence

from numba import njit
from numpy import empty
from numpy.typing import NDArray

from multikeydict.nestedmkdict import NestedMKDict


def refine_detector_data(
    source: NestedMKDict,
    target: NestedMKDict,
    *,
    detectors: Sequence[str],
    periods: Sequence[str] = ("6AD", "8AD", "7AD"),
    clean_source: bool = True,
) -> None:
    fields = ("livetime", "eff", "efflivetime")
    target["days"] = (days_storage := {})
    for det in detectors:
        day = source["day", det]
        step = day[1:] - day[:-1]
        assert (step==1).all(), "Expect detector data for each day"

        ndet = source["ndet", det]
        for periodname in periods:
            period_ndet = int(periodname[0])
            mask_period = ndet == period_ndet

            for field in fields:
                data = source[field, det]

                key = (field, periodname, det)
                target[key] = data[mask_period]

            days = source["day", det][mask_period]
            days_stored = days_storage.setdefault(periodname, days)
            if days is not days_stored:
                assert all(days==days_stored)

    if clean_source:
        for key in tuple(source.walkkeys()):
            source.delete_with_parents(key)


def refine_detector_data2(
    source: NestedMKDict,
    target: NestedMKDict,
    *,
    detectors: Sequence[str],
    clean_source: bool = True,
) -> None:
    fields = ("livetime", "eff", "efflivetime")
    target["days"] = (days_storage := {})
    for det in detectors:
        day = source["day", det]
        step = day[1:] - day[:-1]
        assert (step==1).all(), "Expect detector data for each day"

        for field in fields:
            data = source[field, det]

            key = (field, det)
            target[key] = data

        days = source["day", det]
        days_stored = days_storage.setdefault(det, days)
        if days is not days_stored:
            assert all(days==days_stored)

    if clean_source:
        for key in tuple(source.walkkeys()):
            source.delete_with_parents(key)
