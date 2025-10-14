from psana.detector.detector_impl import DetectorImpl

class generic_container_xtc1dump_0_1_0(DetectorImpl):
    """A generic container to read data dumped from XTC1 into XTC2."""

    def __init__(self, *args):
        super().__init__(*args)

        self._add_fields()
