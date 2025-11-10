# TODO:
# - Keep timestamp callback
# - Adding EpicStore and ScanStore
from psana import DataSource
import os
import pathlib

from psana.psexp.tools import mode
from setup_input_files import setup_input_files

# Check that we use only one bigdata core for testing due to assert below
if mode == 'mpi':
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    assert size in (1, 3), "Test test only allow on bigdata core (try again without mpirun or with -n 3)"
else:
    rank = 0
    size = 1

def smd_callback(run):
    edet = run.Detector('HX2:DVD:GCC:01:PMON,hello1')
    _ = run.Detector('motor2')
    for evt in run.events():
        if evt.timestamp % 2 == 0 and edet(evt) is not None:
            yield evt

def smd_callback_with_step(run):
    edet = run.Detector('HX2:DVD:GCC:01:PMON,hello1')
    _ = run.Detector('motor2')
    for i_step, step in enumerate(run.steps()):
        for i_evt, evt in enumerate(step.events()):
            if evt.timestamp % 2 == 0 and edet(evt) is not None:
                yield evt

def generate_testdata(xtc_path):
    print("Generate test_data")
    if not xtc_path.exists():
        if rank == 0:
            xtc_path.mkdir()
            setup_input_files(xtc_path, n_files=2, slow_update_freq=4, n_motor_steps=3, n_events_per_step=10, gen_run2=False)
            print(f"Done generating data in {xtc_path}")

def run_test_loop_callback(xtc_dir, withstep=False):
    if withstep:
        print('smd_callback_with_step')
        callback = smd_callback_with_step
    else:
        callback = smd_callback

    ds = DataSource(exp='xpptut15', run=14, dir=xtc_dir,
            smd_callback=callback
            )

    cn_events = 0
    for run in ds.runs():
        det = run.Detector('xppcspad')
        edet = run.Detector('HX2:DVD:GCC:01:PMON,hello1')
        sdet = run.Detector('motor2')
        for evt in run.events():
            img = det.raw.calib(evt)
            cn_events +=1
            print(f'bigdata {cn_events=}', evt.timestamp, img.shape, edet(evt), sdet(evt))

    # Only checking bigdata rank
    if ds.unique_user_rank():
        assert cn_events == 13, f"{cn_events=} expected 13"


if __name__ == "__main__":
    root_dir = os.environ.get("TEST_XTC_DIR", ".")
    p = pathlib.Path(os.path.join(root_dir, "tmp_data_for_run_loop_callback"))
    generate_testdata(p)
    xtc_dir = os.path.join(str(p), ".tmp")
    run_test_loop_callback(xtc_dir)
    run_test_loop_callback(xtc_dir, withstep=True)
