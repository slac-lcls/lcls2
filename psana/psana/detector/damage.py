"""
Take the detector segments and manage xtc._damage fields.

Note that each level of xtc class has _damage property:
  d._xtc
  d.xppcspad[0]._xtc
  d.xppcspad[0].raw._xtc
, but we only make the alg level (raw, fex, etc.) xtc._damage
available through the interface below.
"""

from dataclasses import dataclass
from enum import Enum

DAMAGE_USERBITSHIFT = 12
DAMAGE_VALUEBITMASK = 0x0FFF


class DamageBitmask(Enum):
    Truncated = 1  # bitmask 0
    OutOfOrder = 2
    OutOfSynch = 3
    Corrupted = 4
    DroppedContribution = 5
    MissingData = 6
    TimedOut = 7
    UserDefined = 8  # bitmask 12

    def size(self):
        return len(self.__dict__["_member_names_"])


@dataclass
class DamageInfo:
    counts: dict = None
    userbits: list = None
    evt: object = None
    sum_recorded: bool = False

    def loaded(self, evt):
        return self.evt is evt


class Damage:
    def __init__(self, det_alg):
        self.det_alg = det_alg
        self.segments = det_alg._segments
        self._damage_info = DamageInfo()
        self._sum_damage_counts = {}

    def __call__(self, evt):
        return self.count(evt)

    def _expected_segments(self):
        return list(getattr(self.det_alg, "_sorted_segment_inds", []))

    def _damage_vector(self, segment_ids):
        if not segment_ids:
            return []
        return [0] * (max(segment_ids) + 1)

    def _event_damage(self, evt):
        damage = 0
        for dgram in getattr(evt, "_dgrams", []):
            if dgram is None or not hasattr(dgram, "_xtc"):
                continue
            damage |= dgram._xtc.damage & DAMAGE_VALUEBITMASK
        return damage

    def _evt_segments(self, evt):
        det_name = getattr(self.det_alg, "_det_name", None)
        drp_class_name = getattr(self.det_alg, "_drp_class_name", None)
        if det_name is None or drp_class_name is None:
            return {}
        return getattr(evt, "_det_segments", {}).get((det_name, drp_class_name), {})

    def _add_damage(self, damage_counts, damage, segment_id, segment_ids):
        damage &= DAMAGE_VALUEBITMASK
        if not damage:
            return
        if damage not in damage_counts:
            damage_counts[damage] = self._damage_vector(segment_ids)
        damage_counts[damage][segment_id] = 1

    def _add_to_sum(self):
        for damage, segment_counts in self._damage_info.counts.items():
            if damage not in self._sum_damage_counts:
                self._sum_damage_counts[damage] = list(segment_counts)
            else:
                self._sum_damage_counts[damage] = [
                    sum_count + evt_count
                    for sum_count, evt_count in zip(
                        self._sum_damage_counts[damage], segment_counts
                    )
                ]
        self._damage_info.sum_recorded = True

    def _load_damage_info(self, evt, flag_sum=False):
        if self._damage_info.loaded(evt):
            if flag_sum and not self._damage_info.sum_recorded:
                self._add_to_sum()
            return

        "Get segments of det/alg for this event and return damage info"
        segments = self.segments(evt)

        # Idea 1: damage_counts can be global and we only keep the sum
        expected_segments = self._expected_segments()
        segment_ids = expected_segments
        evt_segments = segments if segments is not None else self._evt_segments(evt)
        if evt_segments:
            segment_ids = sorted(set(segment_ids) | set(evt_segments.keys()))
        damage_counts = {}
        userbits = self._damage_vector(segment_ids)

        present_segments = set()
        if evt_segments:
            for segment_id, segment in evt_segments.items():
                present_segments.add(segment_id)
                if hasattr(segment, "_xtc") and segment._xtc.damage:
                    userbits[segment_id] = segment._xtc.damage >> DAMAGE_USERBITSHIFT
                    self._add_damage(
                        damage_counts, segment._xtc.damage, segment_id, segment_ids
                    )

        missing_segments = set(expected_segments) - present_segments
        if missing_segments:
            damage = self._event_damage(evt)
            if not damage and present_segments:
                damage = 1 << DamageBitmask.MissingData.value
            if damage:
                for segment_id in missing_segments:
                    self._add_damage(damage_counts, damage, segment_id, segment_ids)

        self._damage_info = DamageInfo(damage_counts, userbits, evt)
        if flag_sum:
            self._add_to_sum()

    def count(self, evt, flag_sum=True):
        self._load_damage_info(evt, flag_sum=flag_sum)
        return self._damage_info.counts

    def userbits(self, evt):
        self._load_damage_info(evt)
        return self._damage_info.userbits

    def sum(self):
        return self._sum_damage_counts
