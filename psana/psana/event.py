from psana import dgram
from psana.psexp.packet_footer import PacketFooter

class Event():
    """
    Event holds list of dgrams
    """
    def __init__(self, dgrams=[], size=0):
        if size:
            self.dgrams = [0] * size
            self.offsets = [0] * size
            self.size = size
        else:
            self.dgrams = dgrams
            self.offsets = [_d._offset for _d in self.dgrams]
            self.size = len(dgrams)
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.position >= len(self.dgrams):
            raise StopIteration
        event = self.dgrams[self.position]
        self.position += 1
        return event

    def replace(self, pos, d):
        assert pos < self.size
        self.dgrams[pos] = d

    def to_bytes(self):
        event_bytes = bytearray()
        pf = PacketFooter(self.size)
        for i, d in enumerate(self.dgrams):
            event_bytes.extend(bytearray(d))
            pf.set_size(i, memoryview(bytearray(d)).shape[0])

        if event_bytes:
            event_bytes.extend(pf.footer)

        return event_bytes

    def from_bytes(self, configs, event_bytes):
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
    def seconds(self):
        _high = (self.dgrams[0].seq.timestamp() >> 32) & 0xffffffff
        return _high
    
    @property
    def nanoseconds(self):
        _low = self.dgrams[0].seq.timestamp() & 0xffffffff
        return _low

    def run(self):
        return 0 # for psana1-cctbx compatibility

