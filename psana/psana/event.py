# import detectors

"""Define `Event`, the object that holds the dgrams making up one event.

Entries in the dgram list can be None when a dgram is missing from the event.
"""
import datetime
import numpy as np


from psana.psexp.packet_footer import PacketFooter
from psana import utils

# TO DO
# 1) remove comments
# 2) pass detector class table from run > dgrammgr > event
# 3) hook up the detector class table

epoch = datetime.datetime(1990, 1, 1)


class DrpClassContainer(object):
    """Empty placeholder class.

    Its `__init__` body is `pass` and it defines no other members.
    """
    def __init__(self):
        pass


class Event:
    """
    Event holds list of dgrams
    """

    def __init__(self, dgrams, run=None, proxy_evt=None):
        self._dgrams = dgrams
        self._size = len(dgrams)
        self._complete()
        self._position = 0
        self._run = run  # RunCtx object
        self._proxy_evt = proxy_evt # For smalldata-event loop

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    # we believe this can be hidden with underscores when we eliminate py2 support
    def next(self):
        """Return the next dgram of this event and advance the iterator position.

        The returned entry can be None for a missing dgram.

        Raises
        ------
        StopIteration
            When all dgrams have been returned.
        """
        if self._position >= len(self._dgrams):
            raise StopIteration
        d = self._dgrams[self._position]
        self._position += 1
        return d

    def _replace(self, pos, d):
        assert pos < self._size
        self._dgrams[pos] = d

    def _to_bytes(self):
        event_bytes = bytearray()
        pf = PacketFooter(self._size)
        for i, d in enumerate(self._dgrams):
            event_bytes.extend(bytearray(d))
            pf.set_size(i, memoryview(bytearray(d)).shape[0])

        if event_bytes:
            event_bytes.extend(pf.footer)

        return event_bytes

    def timestamp_diff(self, ts):
        """Subtract the given timestamp from the event timestamp and return the difference in nanoseconds.

        A timestamp packs seconds in its upper 32 bits and nanoseconds in its lower 32 bits, so both
        parts of each timestamp are converted to a total number of nanoseconds before subtracting.
        """
        evt_sec = self._seconds
        evt_nsec = self._nanoseconds
        ts_sec = (ts >> 32) & 0xFFFFFFFF
        ts_nsec = ts & 0xFFFFFFFF
        dif_ns = (evt_sec * 1000000000 + evt_nsec) - (ts_sec * 1000000000 + ts_nsec)
        return dif_ns

    @property
    def _seconds(self):
        _high = (self.timestamp >> 32) & 0xFFFFFFFF
        return _high

    @property
    def _nanoseconds(self):
        _low = self.timestamp & 0xFFFFFFFF
        return _low

    @property
    def timestamp(self):
        """Timestamp of the first dgram in the event that provides one.

        Delegates to `psana.utils.first_timestamp`, which skips None dgrams and raises RuntimeError
        if no dgram has a timestamp. `timestamp_diff` and `datetime` treat the upper 32 bits as
        seconds and the lower 32 bits as nanoseconds.
        """
        return utils.first_timestamp(self._dgrams)

    @property
    def env(self):
        """Value of `env()` of the first dgram that provides it.

        Delegates to `psana.utils.first_env`, which skips None dgrams and raises RuntimeError if no
        dgram can provide `env()`.
        """
        return utils.first_env(self._dgrams)

    def run(self):
        """Return the run object given as `run` when the event was created, or None if none was given."""
        return self._run

    def _assign_det_segments(self):
        """ """
        self._det_segments = {}
        for evt_dgram in self._dgrams:

            if evt_dgram:  # dgram can be None (missing) in an event

                # detector name (e.g. "xppcspad")
                for det_name, segment_dict in evt_dgram.__dict__.items():
                    # skip hidden dgram attributes
                    if det_name.startswith("_"):
                        continue

                    # drp class name (e.g. "raw", "fex")
                    for segment, det in segment_dict.items():
                        for drp_class_name, drp_class in det.__dict__.items():
                            class_identifier = (det_name, drp_class_name)

                            if class_identifier not in self._det_segments.keys():
                                self._det_segments[class_identifier] = {}
                            segs = self._det_segments[class_identifier]

                            if det_name not in ["runinfo", "smdinfo", "chunkinfo"]:
                                assert (
                                    segment not in segs
                                ), f"Found duplicate segment: {segment} for {class_identifier}"
                            segs[segment] = drp_class

        return

    # this routine is called when all the dgrams have been inserted into
    # the event (e.g. by the eventbuilder calling _replace())
    def _complete(self):
        self._assign_det_segments()

    @property
    def _has_offset(self):
        return hasattr(self._dgrams[0], "info")

    def service(self):
        """Return the service code of the event.

        Delegates to `psana.utils.first_service`, which returns `(env() >> 24) & 0xF` of the first
        valid dgram and raises RuntimeError if that value is outside 1..13 or no dgram provides
        `env()`.
        """
        return utils.first_service(self._dgrams)

    def keepraw(self):
        """Return the keepraw bit of the event.

        Returns
        -------
        int or None
            Bit 22 of `env()` of the first non-None dgram (0 or 1), or None if every dgram is None.
        """
        keepraw = None
        for d in self._dgrams:
            if d:
                keepraw = (d.env() >> 22) & 0x1
                break
        return keepraw

    def get_offsets_and_sizes(self):
        """Return the offset and size of every dgram in the event.

        Returns
        -------
        numpy.ndarray
            int64 array of shape (number of dgrams, 2); row i is `get_offset_and_size(i)`: offset and
            dgram size from the `smdinfo` field when the dgram has one, otherwise offset 0 and the
            dgram's `_size`, and 0, 0 for a missing (None) dgram.
        """
        offset_and_size_arr = np.zeros((self._size, 2), dtype=np.int64)
        for i in range(self._size):
            offset_and_size_arr[i, :] = self.get_offset_and_size(i)
        return offset_and_size_arr

    def get_offset_and_size(self, i):
        """
        Returns value of offset and size stored in smdinfo event
        * For other event, return 0 of offset and size of the dgram
        for size.
        """
        d = self._dgrams[i]
        offset_and_size = np.zeros((1, 2), dtype=int)
        if d:
            if hasattr(d, "smdinfo"):
                offset_and_size[:] = [
                    d.smdinfo[0].offsetAlg.intOffset,
                    d.smdinfo[0].offsetAlg.intDgramSize,
                ]
            else:
                offset_and_size[0, 1] = d._size
        return offset_and_size

    def datetime(self):
        """Return the event timestamp as a `datetime.datetime`.

        Adds the seconds (upper 32 bits) and the nanoseconds (lower 32 bits, divided by 1000 to give
        microseconds) to the epoch 1990-01-01 00:00:00. The result is a naive datetime (no time zone).
        """
        sec = self.timestamp >> 32
        usec = (self.timestamp & 0xFFFFFFFF) / 1000
        delta_t = datetime.timedelta(seconds=sec, microseconds=usec)
        return epoch + delta_t

    def set_destination(self, dest):
        """Sets destination (bigdata core rank no.) where this event
        should be sent to.

        Destination only works in parallel mode with only one EvenBuilder core
        (PS_EB_NODES=1). The valid range of destination is from 1 to the number
        of available bigdata cores.
        """
        self._proxy_evt.set_destination(dest)

    def EndOfBatch(self):
        """Returns a list of integrating detectors whose batch ends at this event

        Single integrating detector:
            defaulted to [intg_det]
        Multiple integrating detectors (future feature):
            we probably need another flag to identify which integrating detector(s)
            marks the end of this batch.
        """
        intg_dets = []
        for d in self._dgrams:
            if hasattr(d, "_endofbatch"):
                intg_dets.append(self._run.intg_det)
        return intg_dets
