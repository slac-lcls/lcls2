class Event:
    """
    Event holds list of dgrams
    """
    def __init__(self, dgrams=[]):
        self.dgrams = dgrams
        self.offsets = [_d._offset for _d in self.dgrams]
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.position >= len(self.dgrams):
            raise StopIteration
        event = self.dgrams[self.position]
        self.position += 1
        return event

