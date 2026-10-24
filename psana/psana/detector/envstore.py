"""Detector interfaces for env-store variables (epics and scan): `EnvImpl` and its version-named subclasses."""
from psana.detector.detector_impl import DetectorImpl
from psana.event import Event
from psana.psexp.step import Step

class InvalidInputEnvStore(Exception):
    """Exception raised by `EnvImpl.__call__` when its argument is not a list, an `Event` or a `Step`."""
    pass

class EnvImpl(DetectorImpl):
    """Detector interface for one env-store variable (an epics or scan variable).

    Calling the object with an `Event`, a `Step` (its step event is used) or a list of events returns
    the variable's value, or a list of values for a list, from `EnvStore.values` (None where no
    value is found). If the object has no env store the call executes `raise MissingEnvStore(...)`,
    but `MissingEnvStore` is not defined or imported in this module, so a NameError results.
    """
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
        """Type recorded in the env store for this variable (`EnvStore.dtype(var_name)`), or None if it is not found."""
        return self._env_store.dtype(self._var_name)

class epics_epics_0_0_0(EnvImpl):
    """`EnvImpl` for detector type `epics`, software `epics`, version 0.0.0; adds nothing to `EnvImpl`."""
    def __init__(self, *args):
        super().__init__(*args)

class epics_epics_0_0_1(EnvImpl):
    """`EnvImpl` for detector type `epics`, software `epics`, version 0.0.1; adds nothing to `EnvImpl`."""
    def __init__(self, *args):
        super().__init__(*args)

class epics_raw_2_0_0(EnvImpl):
    """`EnvImpl` for detector type `epics`, software `raw`, version 2.0.0; adds nothing to `EnvImpl`."""
    def __init__(self, *args):
        super().__init__(*args)

class epics_opaltt_2_0_0(EnvImpl):
    """`EnvImpl` for detector type `epics`, software `opaltt`, version 2.0.0; adds nothing to `EnvImpl`."""
    def __init__(self, *args):
        super().__init__(*args)

class epics_piranha4tt_1_0_0(EnvImpl):
    """`EnvImpl` for detector type `epics`, software `piranha4tt`, version 1.0.0; adds nothing to `EnvImpl`."""
    def __init__(self, *args):
        super().__init__(*args)

class scanDet_scan_2_3_42(EnvImpl):
    """`EnvImpl` for detector type `scanDet`, software `scan`, version 2.3.42; adds nothing to `EnvImpl`."""
    def __init__(self, *args):
        super().__init__(*args)

class scan_raw_2_0_0(EnvImpl):
    """`EnvImpl` for detector type `scan`, software `raw`, version 2.0.0; adds nothing to `EnvImpl`."""
    def __init__(self, *args):
        super().__init__(*args)

class scan_raw_1_0_0(EnvImpl):
    """`EnvImpl` for detector type `scan`, software `raw`, version 1.0.0; adds nothing to `EnvImpl`."""
    def __init__(self, *args):
        super().__init__(*args)

class scan_raw_0_0_2(EnvImpl):
    """`EnvImpl` for detector type `scan`, software `raw`, version 0.0.2; adds nothing to `EnvImpl`."""
    def __init__(self, *args):
        super().__init__(*args)
