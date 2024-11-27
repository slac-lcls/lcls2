from dataclasses import dataclass

DAMAGE_USERBITSHIFT = 12
DAMAGE_VALUEBITMASK = 0x0FFF


@dataclass
class DamageInfo:
    Counts: dict = None
    userBits: list = None


class Damage:
    def __init__(self, segments):
        self.segments = segments

    def __call__(self, evt):
        return self._get_damage_info(evt)

    def _get_damage_info(self, evt):
        "Get segments of det/alg for this event and return damage info"
        segments = self.segments(evt)
        if segments is None:
            return DamageInfo()

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
        damage_info = DamageInfo(damage_counts, userbits)
        return damage_info

    def Count(self, evt):
        return self._get_damage_info(evt).Counts

    def userBits(self, evt):
        return self._get_damage_info(evt).userBits
