from psana import DataSource 
import os

# - Keep timestamp callback
# - Eliminate destination and filter_fn
# - Adding EpicStore and ScanStore
def smd_callback(smd_ds):
    for i, evt in enumerate(smd_ds.events()):
        #smd_ds.set_destination(evt, rank=3)
        if evt.timestamp % 2 == 0:
            yield evt

def run_test_loop_callback():
    xtc_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.tmp')
    ds = DataSource(exp='xpptut15', run=1, dir=xtc_dir,
            smd_callback=smd_callback
            )
    for run in ds.runs():
        det = run.Detector('xppcspad')
        edet = run.Detector('HX2:DVD:GCC:01:PMON,hello1')
        sdet = run.Detector('motor2')
        for evt in run.events():
            img = det.raw.calib(evt)
            print('bigdata', evt.timestamp, img.shape, sdet(evt), edet(evt))


if __name__ == "__main__":
    run_test_loop_callback()
            
