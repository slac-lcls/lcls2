import os
import sys

from mpi4py import MPI

from psana import DataSource

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


os.environ["PS_SRV_NODES"] = "1"
os.environ["PS_SMD_N_EVENTS"] = "1"


# passing exp and runnum
exp = sys.argv[1]
runnum = int(sys.argv[2])


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
    timing = myrun.Detector("timing")
    smd = ds.smalldata(
        filename="mysmallh5.h5",
        batch_size=5,
    )
    for nevt, evt in enumerate(myrun.events()):
        smd.event(evt, mydata=nevt)
        if nevt % 2 == 0:
            smd.event(evt, grp_mygroup_a=45.1)
            # alternative interface 1
            # smd.event(evt, a=45.1, group="mygroup")
            # atlernatvie interface 2
            # smd.event(evt, {"mygroup": {a: 45.1, b: }}
        if nevt % 5 == 0:
            smd.event(
                evt,
                mydata=nevt,
                sum_x=42.2,
                unaligned_y=43,
                grp_mygroup_b=46,
                grp_anothergroup_a=100,
            )

smd.done()
