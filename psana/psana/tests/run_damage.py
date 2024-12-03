from psana import DataSource
from psana.detector import Damage

ds = DataSource(exp="rixx1016923", run=26, max_events=10)
myrun = next(ds.runs())
det = myrun.Detector("gmd")
damage = Damage(det.raw)  # use this interface to hide it from ami

for evt in myrun.events():
    evt_damage_counts = damage.count(evt)  # this line is required to compute the sum
    evt_damage_userbits = damage.userbits(evt)
    print(evt_damage_counts)
    print(evt_damage_userbits)

# Both sum and evt _damage_counts variables has DamageName as the key
# sum_damage_counts = damage.sum()  # can be called while looping over the events

# d._xtc
# d.xppcspad[0]._xtc
# d.xppcspad[0].raw._xtc
