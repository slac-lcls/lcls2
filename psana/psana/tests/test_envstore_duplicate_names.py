from psana.detector.envstore import AmbiguousEnvStoreDetector, InvalidInputEnvStore
from psana.event import Event
from psana.psexp.envstore_manager import EnvStoreManager
from psana.psexp.step import Step


class _FakeStore:
    def __init__(self, mapping):
        self._mapping = mapping

    def locate_variable(self, name):
        return self._mapping.get(name)


class _FakeDet:
    def __init__(self, label):
        self.label = label
        self.dtype = f"dtype-{label}"
        self.calls = []

    def __call__(self, events):
        self.calls.append(events)
        return self.label


def test_envstore_manager_reports_all_matches():
    esm = EnvStoreManager.__new__(EnvStoreManager)
    esm.stores = {
        "epics": _FakeStore({"lxt": ("raw", 0)}),
        "scan": _FakeStore({"lxt": ("raw", 0)}),
    }

    assert esm.envs_from_variable("lxt") == [("epics", "raw"), ("scan", "raw")]
    assert esm.env_from_variable("lxt") == ("epics", "raw")
    assert esm.envs_from_variable("missing") == []
    assert esm.env_from_variable("missing") is None


def test_ambiguous_env_detector_dispatches_by_input_type():
    epics = _FakeDet("epics")
    scan = _FakeDet("scan")
    det = AmbiguousEnvStoreDetector("lxt", {"epics": epics, "scan": scan})

    evt = Event.__new__(Event)
    step = Step.__new__(Step)

    assert det(evt) == "epics"
    assert det(step) == "scan"
    assert det([evt]) == "epics"
    assert det([step]) == "scan"
    assert det.dtype == "dtype-epics"

    # Explicit access keeps both stores reachable when names overlap.
    assert det.epics(evt) == "epics"
    assert det.scan(step) == "scan"


def test_ambiguous_env_detector_rejects_invalid_input():
    det = AmbiguousEnvStoreDetector("lxt", {"epics": _FakeDet("epics")})
    try:
        det(1.23)
    except InvalidInputEnvStore:
        pass
    else:
        assert False, "Expected InvalidInputEnvStore"
