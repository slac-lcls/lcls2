from psana.detector.detector_impl import DetectorImpl
from psana.event import Event
from psana.psexp.step import Step

class InvalidInputEnvStore(Exception): pass

class EnvImpl(DetectorImpl):
    def __init__(self, *args):
        super(EnvImpl, self).__init__(*args)
    
    def __call__(self, events):
        if self._env_store is None:
            err_msg = f"Function call is not available for this detector."
            raise MissingEnvStore(err_msg)

        if isinstance(events, list):
            return self._env_store.values(events, self._var_name)
        elif isinstance(events, Event):
            env_values = self._env_store.values([events], self._var_name)
            return env_values[0]
        elif isinstance(events, Step):
            env_values = self._env_store.values([events.evt], self._var_name)
            return env_values[0]
        err_msg = f"Calling detector only accept Event or Step. Invalid type: {type(events)} given."
        raise InvalidInputEnvStore(err_msg)
    
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

class epics_opaltt_2_0_0(EnvImpl):
    def __init__(self, *args):
        super().__init__(*args)

class scanDet_scan_2_3_42(EnvImpl):
    def __init__(self, *args):
        super().__init__(*args)

class scan_raw_2_0_0(EnvImpl):
    def __init__(self, *args):
        super().__init__(*args)

