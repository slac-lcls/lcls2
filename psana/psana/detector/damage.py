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

    def loaded(self):
        return self.counts


class Damage:
    def __init__(self, det_alg):
        self.segments = det_alg._segments
        self._damage_info = DamageInfo()
        self._sum_damage_counts = {}

    def __call__(self, evt):
        return self._get_damage_info(evt)

    def _load_damage_info(self, evt, flag_sum=False):
        if self._damage_info.loaded():
            return

        "Get segments of det/alg for this event and return damage info"
        segments = self.segments(evt)
        if segments is None:
            return

        # Idea 1: damage_counts can be global and we only keep the sum
        damage_counts = {}
        userbits = [0] * len(segments)
        for segment_id, segment in segments.items():
            if hasattr(segment, "_xtc"):
                if segment._xtc.damage:
                    userbits[segment_id] = segment._xtc.damage >> DAMAGE_USERBITSHIFT
                    damageId = segment._xtc.damage & DAMAGE_VALUEBITMASK
                    if damageId in damage_counts:
                        damage_counts[damageId][segment_id] = 1
                    else:
                        damage_counts[damageId] = [
                            0 if i != segment_id else 1 for i in range(len(segments))
                        ]
                    # Calculate accumulated sum for the damage counts
                    if flag_sum:
                        for damageId, seg_damage in damage_counts.items():
                            if damageId not in self._sum_damage_counts:
                                self._sum_damage_counts[damageId] = seg_damage
                            else:
                                self._sum_damage_counts[damageId] += seg_damage

        self._damage_info = DamageInfo(damage_counts, userbits)

    def count(self, evt, flag_sum=False):
        self._load_damage_info(evt, flag_sum=flag_sum)
        return self._damage_info.counts

    def userbits(self, evt):
        self._load_damage_info(evt)
        return self._damage_info.userbits

    def sum(self):
        return self._sum_damage_counts
