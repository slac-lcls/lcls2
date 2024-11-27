from psana import DataSource

ds = DataSource(exp="rixx1016923", run=26)
myrun = next(ds.runs())
det = myrun.Detector("gmd")

for evt in myrun.events():
    damage_counts = det.raw.Damage.Count(evt)
    if damage_counts:
        print(evt.timestamp, damage_counts)
