Truncated = 1  # bitmask 0
OutOfOrder = 2
OutOfSynch = 4
Corrupted = 16
DroppedContribution = 32
MissingData = 64
TimedOut = 128
UserDefined = 4096  # bitmask 12

damage_id_to_name = {
    1: "Truncated",
    2: "OutOfOrder",
    4: "OutOfSynch",
    16: "Corrupted",
    32: "DroppedContribution",
    64: "MissingData",
    128: "TimedOut",
    4096: "UserDefined",
}


def damageName(damageId):
    return damage_id_to_name[damageId]
