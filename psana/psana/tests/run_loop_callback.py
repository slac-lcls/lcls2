from psana import DataSource 
import os

# TODO:
# - Keep timestamp callback
# - Eliminate destination and filter_fn
# - Adding EpicStore and ScanStore
def smd_callback(run):
    edet = run.Detector('HX2:DVD:GCC:01:PMON,hello1')
    sdet = run.Detector('motor2')
    for i_evt, evt in enumerate(run.events()):
        if evt.timestamp % 2 == 0 and edet(evt) is not None:
            yield evt

def smd_callback_with_step(run):
    edet = run.Detector('HX2:DVD:GCC:01:PMON,hello1')
    sdet = run.Detector('motor2')
    for i_step, step in enumerate(run.steps()):
        for i_evt, evt in enumerate(step.events()):
            if evt.timestamp % 2 == 0 and edet(evt) is not None:
                yield evt
                

def run_test_loop_callback(withstep=False):
    xtc_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.tmp_smd0', '.tmp')

    if withstep:
        print(f'smd_callback_with_step')
        callback = smd_callback_with_step
    else:
        callback = smd_callback

    ds = DataSource(exp='xpptut15', run=1, dir=xtc_dir,
            smd_callback=callback
            )

    cn_events = 0
    for run in ds.runs():
        det = run.Detector('xppcspad')
        edet = run.Detector('HX2:DVD:GCC:01:PMON,hello1')
        sdet = run.Detector('motor2')
        for evt in run.events():
            img = det.raw.calib(evt)
            print('bigdata', evt.timestamp, img.shape, edet(evt), sdet(evt))
            cn_events +=1

    assert cn_events == 13


if __name__ == "__main__":
    run_test_loop_callback()
    run_test_loop_callback(withstep=True)
            
