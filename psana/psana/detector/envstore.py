from psana.detector.detector_impl import DetectorImpl
from psana.event import Event
from psana.psexp.step import Step

class InvalidInputEnvStore(Exception): pass


class AmbiguousEnvStoreDetector:
    """Dispatches duplicated env variable names to the appropriate store.

    For variables that exist in more than one env store (currently scan/epics),
    this wrapper routes:
      - Step input to scan (if available)
      - Event input to epics (if available)
    """

    def __init__(self, var_name, detectors_by_env):
        self._var_name = var_name
        self._detectors_by_env = detectors_by_env
        for env_name, det in detectors_by_env.items():
            setattr(self, env_name, det)

        self._event_env = "epics" if "epics" in detectors_by_env else next(
            iter(detectors_by_env)
        )
        self._step_env = "scan" if "scan" in detectors_by_env else self._event_env

    def _select_detector(self, events):
        if isinstance(events, Step):
            return self._detectors_by_env[self._step_env]
        if isinstance(events, Event):
            return self._detectors_by_env[self._event_env]
        if isinstance(events, list):
            if not events:
                return self._detectors_by_env[self._event_env]
            if isinstance(events[0], Step):
                return self._detectors_by_env[self._step_env]
            if isinstance(events[0], Event):
                return self._detectors_by_env[self._event_env]
        err_msg = f"Calling detector only accept Event or Step. Invalid type: {type(events)} given."
        raise InvalidInputEnvStore(err_msg)

    def __call__(self, events):
        detector = self._select_detector(events)
        return detector(events)

    @property
    def dtype(self):
        return self._detectors_by_env[self._event_env].dtype


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

class epics_piranha4tt_1_0_0(EnvImpl):
    def __init__(self, *args):
        super().__init__(*args)

class scanDet_scan_2_3_42(EnvImpl):
    def __init__(self, *args):
        super().__init__(*args)

class scan_raw_2_0_0(EnvImpl):
    def __init__(self, *args):
        super().__init__(*args)

class scan_raw_1_0_0(EnvImpl):
    def __init__(self, *args):
        super().__init__(*args)

class scan_raw_0_0_2(EnvImpl):
    def __init__(self, *args):
        super().__init__(*args)
