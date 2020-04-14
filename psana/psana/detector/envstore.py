from psana.detector.detector_impl import DetectorImpl

class EnvImpl(DetectorImpl):
    def __init__(self, *args):
        det_name, self._var_name, drp_class_name, configs, calibs, self._env_store = args
        super(EnvImpl, self).__init__(det_name, drp_class_name, configs, calibs)
    def __call__(self, events):
        if isinstance(events, list):
            return self._env_store.values(events, self._var_name)
        else:
            env_values = self._env_store.values([events], self._var_name)
            return env_values[0]
    @property
    def dtype(self):
        return self._env_store.dtype(self._var_name)


class epics_epics_0_0_0(EnvImpl):
    def __init__(self, *args):
        super().__init__(*args)

class epics_epics_0_0_1(EnvImpl):
    def __init__(self, *args):
        super().__init__(*args)

class epics_raw_2_0_0(EnvImpl):
    def __init__(self, *args):
        super().__init__(*args)

class scanDet_scan_2_3_42(EnvImpl):
    def __init__(self, *args):
        super().__init__(*args)

class scan_raw_2_0_0(EnvImpl):
    def __init__(self, *args):
        super().__init__(*args)

