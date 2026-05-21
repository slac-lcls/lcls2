import importlib.util
from pathlib import Path
from types import SimpleNamespace


_DAMAGE_PATH = Path(__file__).resolve().parents[1] / "detector" / "damage.py"
_SPEC = importlib.util.spec_from_file_location("damage_under_test", _DAMAGE_PATH)
_DAMAGE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DAMAGE)

DAMAGE_USERBITSHIFT = _DAMAGE.DAMAGE_USERBITSHIFT
Damage = _DAMAGE.Damage
DamageBitmask = _DAMAGE.DamageBitmask


class _FakeEvent:
    pass


class _FakeDetAlg:
    def __init__(self, evt_to_segments, expected_segments=None):
        self._evt_to_segments = evt_to_segments
        self._sorted_segment_inds = expected_segments or []
        self._det_name = "det"
        self._drp_class_name = "raw"

    def _segments(self, evt):
        return self._evt_to_segments.get(evt)


def _segment(damage=0):
    return SimpleNamespace(_xtc=SimpleNamespace(damage=damage))


def _event(dgram_damage=0, det_segments=None):
    evt = _FakeEvent()
    evt._dgrams = [SimpleNamespace(_xtc=SimpleNamespace(damage=dgram_damage))]
    evt._det_segments = {}
    if det_segments is not None:
        evt._det_segments[("det", "raw")] = det_segments
    return evt


def test_damage_count_updates_sum_and_userbits_per_event():
    evt1 = object()
    evt2 = object()
    evt3 = object()
    missing_data = DamageBitmask.MissingData.value
    evt_to_segments = {
        evt1: {
            0: _segment((3 << DAMAGE_USERBITSHIFT) | missing_data),
            1: _segment(0),
        },
        evt2: {
            0: _segment(0),
            1: _segment((1 << DAMAGE_USERBITSHIFT) | missing_data),
        },
        evt3: {
            0: _segment(0),
            1: _segment(0),
        },
    }
    damage = Damage(_FakeDetAlg(evt_to_segments))

    assert damage.count(evt1) == {missing_data: [1, 0]}
    assert damage.userbits(evt1) == [3, 0]
    assert damage.sum() == {missing_data: [1, 0]}

    assert damage.count(evt2) == {missing_data: [0, 1]}
    assert damage.userbits(evt2) == [0, 1]
    assert damage.sum() == {missing_data: [1, 1]}

    assert damage.count(evt3) == {}
    assert damage.userbits(evt3) == [0, 0]
    assert damage.sum() == {missing_data: [1, 1]}


def test_damage_count_does_not_double_count_same_event():
    evt = object()
    corrupted = DamageBitmask.Corrupted.value
    damage = Damage(_FakeDetAlg({evt: {0: _segment(corrupted)}}))

    assert damage.count(evt) == {corrupted: [1]}
    assert damage.count(evt) == {corrupted: [1]}
    assert damage.sum() == {corrupted: [1]}


def test_damage_count_reports_missing_segments_from_event_damage():
    evt = _event(dgram_damage=1 << DamageBitmask.DroppedContribution.value)
    damage = Damage(_FakeDetAlg({evt: None}, expected_segments=[0]))

    assert damage.count(evt) == {1 << DamageBitmask.DroppedContribution.value: [1]}
    assert damage.userbits(evt) == [0]
    assert damage.sum() == {1 << DamageBitmask.DroppedContribution.value: [1]}
