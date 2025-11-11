# import detectors

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
        """Subtract the given timestamp from the event timestamp and return
        differences in nanasecond unit.

        Since timestamp format is [32 bits of seconds][32 bits of nanoseconds],
        we need to convert these two parts in seconds and nanoseconds
        respectively prior to the calculation
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
        return utils.first_timestamp(self._dgrams)

    @property
    def env(self):
        return utils.first_env(self._dgrams)

    def run(self):
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
        return utils.first_service(self._dgrams)

    def keepraw(self):
        keepraw = None
        for d in self._dgrams:
            if d:
                keepraw = (d.env() >> 22) & 0x1
                break
        return keepraw

    def get_offsets_and_sizes(self):
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
