"""Timestamp lookup through recorded EPICS updates and the public detector API."""

import pytest

from psana import DataSource
import psana.dgramCreate as dc
from psana.psexp import TransitionId


@pytest.mark.parametrize(
    "updates, expected",
    [
        ([], [None, None, None, None, None]),
        ([(10, 100), (20, 200)], [None, 100, 100, 200, 200]),
        ([(10, 100), (20, 200), (20, 201)], [None, 100, 100, 201, 201]),
    ],
    ids=["empty-history", "changing-values", "equal-update-timestamps"],
)
def test_epics_values_for_retained_events(tmp_path, updates, expected):
    """Reading later SlowUpdates must not change an earlier event's value."""
    path = tmp_path / "epics.xtc2"
    writer = dc.CyDgram()
    epics = dc.nameinfo("epics", "epics", "test", 0)
    raw = dc.alg("raw", [2, 0, 0])
    timestamps = [5, 10, 15, 20, 25]

    with path.open("wb") as stream:
        # Configure defines the field; its payload is not an EPICS update.
        writer.addDet(epics, raw, {"test_pv": 0})
        stream.write(writer.get(0, TransitionId.Configure))
        stream.write(writer.get(1, TransitionId.BeginRun))
        stream.write(writer.get(2, TransitionId.BeginStep))
        stream.write(writer.get(3, TransitionId.Enable))

        update_index = 0
        for timestamp in timestamps:
            # Equal-time updates precede the L1 event, in their stored order.
            while update_index < len(updates) and updates[update_index][0] <= timestamp:
                update_timestamp, value = updates[update_index]
                writer.addDet(epics, raw, {"test_pv": value})
                stream.write(writer.get(update_timestamp, TransitionId.SlowUpdate))
                update_index += 1
            stream.write(writer.get(timestamp, TransitionId.L1Accept))

        stream.write(writer.get(26, TransitionId.Disable))
        stream.write(writer.get(27, TransitionId.EndStep))
        stream.write(writer.get(28, TransitionId.EndRun))

    ds = DataSource(files=str(path), skip_calib_load="all")
    run = next(ds.runs())
    detector = run.Detector("test_pv")
    events = []
    immediate_values = []
    for event in run.events():
        events.append(event)
        immediate_values.append(detector(event))

    assert [event.timestamp for event in events] == timestamps
    assert immediate_values == expected
    assert [detector(event) for event in events] == expected
    assert detector(events) == expected
