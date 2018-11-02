from psana import dgram
from psana.psexp.packet_footer import PacketFooter

class Event():
    """
    Event holds list of dgrams
    """
    def __init__(self, dgrams=[], size=0):
        if size:
            self._dgrams = [0] * size
            self._offsets = [0] * size
            self._size = size
        else:
            self._dgrams = dgrams
            self._offsets = [_d._offset for _d in self._dgrams]
            self._size = len(dgrams)
        self._position = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self._position >= len(self._dgrams):
            raise StopIteration
        event = self._dgrams[self._position]
        self._position += 1
        return event

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

    def _from_bytes(self, configs, event_bytes):
        dgrams = []
        if event_bytes:
            pf = PacketFooter(view=event_bytes)
            views = pf.split_packets()
            assert len(configs) == len(views)
            dgrams = [dgram.Dgram(config=configs[i], view=views[i]) \
                    for i in range(len(configs))]
        evt = Event(dgrams=dgrams)
        return evt
    
    @property
    def _seconds(self):
        _high = (self._dgrams[0].seq.timestamp() >> 32) & 0xffffffff
        return _high
    
    @property
    def _nanoseconds(self):
        _low = self._dgrams[0].seq.timestamp() & 0xffffffff
        return _low

    def _run(self):
        return 0 # for psana1-cctbx compatibility

