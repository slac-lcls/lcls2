from psana import DataSource
from psana.detector import Damage

ds = DataSource(exp="rixx1016923", run=26, max_events=0)
myrun = next(ds.runs())
det = myrun.Detector("gmd")
damage = Damage(det.raw)  # use this interface to hide it from ami

for evt in myrun.events():
    evt_damage_counts = damage.count(evt)  # this line is required to compute the sum
    evt_damage_userbits = damage.userbits(evt)
    sum_damage_counts = damage.sum() # can be called while looping over the events
    if evt_damage_counts:
        print(f'{evt_damage_counts=}')
        print(f'{evt_damage_userbits=}')
        print(f'{sum_damage_counts=}')
