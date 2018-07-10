from psana import dgram

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
        for i, d in enumerate(self.dgrams):
            event_bytes.extend(bytearray(d))
            if i < self.size - 1:
                event_bytes.extend(b'eod')

        if event_bytes:
            event_bytes.extend(b'endofevt')

        return event_bytes

    def from_bytes(self, configs, event_bytes):
        dgrams = []
        if event_bytes:
            dgrams_bytes = event_bytes.split(b'eod')
            assert len(configs) == len(dgrams_bytes)
            dgrams = [dgram.Dgram(config=configs[i], view=dgrams_bytes[i]) \
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



