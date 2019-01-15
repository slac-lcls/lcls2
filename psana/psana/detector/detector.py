class Detector:
    def __init__(self, name):
        self._name  = name

    def __call__(self, evt):
        return getattr(evt, self._name)
