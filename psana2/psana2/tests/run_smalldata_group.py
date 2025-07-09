# test smalldata group
# run with mpirun -n 4 python run_smalldata_group.py rixc00221 49

import os
import sys
from mpi4py import MPI
from psana2 import DataSource
import h5py

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

os.environ["PS_SRV_NODES"] = "1"
os.environ["PS_SMD_N_EVENTS"] = "1"


OUTPUT_H5FNAME="mysmallh5.h5"
KNOWN_ANSWERS = {"mydata": (10,), "sum_x": (10,), "timestamp": (10,), "mygroup/myval_a": (6,), "mygroup/myval_b": (6,), "mygroup/timestamp": (6,), "another_group/myval_c": (2,), "another_group/timestamp": (2,)}


def check_answers():
    f = h5py.File(OUTPUT_H5FNAME, "r")
    for key, val in KNOWN_ANSWERS.items():
        print(f'{key=} expected:{val} got:{f[key].shape}')
        assert f[key].shape == val
    f.close()


def smd_callback(data_dict):
    print(data_dict.keys())

def run(exp, runnum):
    mount_dir = "/sdf/data/lcls/drpsrcf/ffb"
    xtc_dir = os.path.join(mount_dir, exp[:3], exp, "xtc")
    ds = DataSource(
        exp=exp,
        run=runnum,
        dir=xtc_dir,
        batch_size=1,
        max_events=10,
    )

    for myrun in ds.runs():
        smd = ds.smalldata(
            filename=OUTPUT_H5FNAME,
            batch_size=5,
            callbacks=[smd_callback]
        )
        for nevt, evt in enumerate(myrun.events()):
            smd.event(evt, mydata=nevt)
            if nevt % 2 == 0:
                smd.event(evt, myval_b=42.1, align_group="mygroup")
            if nevt % 5 == 0:
                smd.event(
                    evt,
                    mydata=nevt,
                    sum_x=44,
                )
                smd.event(evt, myval_a=42.0, align_group="mygroup")
                smd.event(evt, myval_c=42.2, align_group="another_group")

    smd.done()

if __name__ == "__main__":
    # passing exp and runnum
    exp = sys.argv[1]
    runnum = int(sys.argv[2])
    run(exp, runnum)
    comm.Barrier()
    if rank == 0:
        check_answers()
