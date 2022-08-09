from psana import DataSource 
import os

def filter_fn(evt):
    if evt.timestamp % 2 == 0:
        return True
    else:
        return False

def smd_callback(smd_ds):
    for i, evt in enumerate(smd_ds.events()):
        if i < 2:
            #smd_ds.set_destination(evt, rank=3)
            yield evt

def run_test_loop_callback():
    xtc_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.tmp')
    ds = DataSource(exp='xpptut15', run=1, dir=xtc_dir,
            filter=filter_fn,
            batch_size=1,
            smd_callback=smd_callback)
    for run in ds.runs():
        det = run.Detector('xppcspad')
        edet = run.Detector('HX2:DVD:GCC:01:PMON,hello1')
        sdet = run.Detector('motor2')
        for evt in run.events():
            img = det.raw.calib(evt)
            print(evt.timestamp, img.shape, edet(evt), sdet(evt))


if __name__ == "__main__":
    run_test_loop_callback()
            
