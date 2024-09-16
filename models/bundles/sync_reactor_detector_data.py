from collections.abc import Sequence

from numba import njit
from numpy import empty
from numpy.typing import NDArray

from multikeydict.nestedmkdict import NestedMKDict


def sync_reactor_detector_data(
    reactor_data: NestedMKDict,
    detector_data: NestedMKDict,
) -> None:
    reactor_day = reactor_data("days")
    detector_day = detector_data("days")

    for period, detector_day_p in detector_day.items():
        reactor_day_p = reactor_day[period]

        offset1 = int(detector_day_p[0] - reactor_day_p[0])
        offset2 = int(detector_day_p[-1] - reactor_day_p[-1])
        if offset1 < 0 or offset2 > 0:
            raise RuntimeError(
                "Reactor data is expected to start not later and end later then detector data. "
                f"{period} got start/end for reactor ({reactor_day_p[0]:.0f}, {reactor_day_p[-1]:.0f})"
                f" and detector ({detector_day_p[0]:.0f}, {detector_day_p[-1]:.0f})"
            )
        if offset2 == 0:
            offset2 = None

        slc = slice(offset1, offset2)
        reactor_day_p_new = reactor_day_p[slc]
        assert (
            reactor_day_p_new == detector_day_p
        ).all(), f"Unable to synchronize reactor and detector data for {period}"

        reactor_day[period] = reactor_day_p_new

        for key, data in reactor_data.walkitems():
            if key[0]=='days':
                continue
            if not period in key:
                continue

            reactor_data[key] = data[slc]
