import sys
from psana import DataSource
import numpy as np
import vals

known_epics_pedestals = np.array([[11.,12.,13.,14.,15.,16.],
     [21.,22.,23.,24.,25.,26.],
     [31.,32.,33.,34.,35.,36.]])

def launch_client(pid, supervisor=-1, supervisor_ip_addr=None):
    dg_count = 0
    if supervisor == -1:
        ds = DataSource(shmem='shmem_test_'+pid, use_calib_cache=True, log_level='DEBUG')
    else:
        ds = DataSource(shmem='shmem_test_'+pid, supervisor=supervisor, supervisor_ip_addr=supervisor_ip_addr, use_calib_cache=True, log_level='DEBUG')

    run = next(ds.runs())

    # Check calibration constant
    assert np.array_equal(ds.dsparms.calibconst['epics']['pedestals'][0], known_epics_pedestals)

    cspad = run.Detector('xppcspad')
    hsd = run.Detector('xpphsd')
    for evt in run.events():
        assert(hsd.raw.calib(evt).shape==(5,))
        assert(hsd.fex.calib(evt).shape==(6,))
        padarray = vals.padarray
        assert(np.array_equal(cspad.raw.calib(evt),np.stack((padarray,padarray))))
        assert(np.array_equal(cspad.raw.image(evt),np.vstack((padarray,padarray))))
        dg_count += 1
    return dg_count

#------------------------------

def main() :
    if len(sys.argv)==2:
        # No pubsub broadcast
        sys.exit(launch_client(sys.argv[1]))
    elif len(sys.argv)==4:
        # Launch in pubsub mode
        sys.exit(launch_client(sys.argv[1], supervisor=int(sys.argv[2]), supervisor_ip_addr=sys.argv[3]))

#------------------------------

if __name__ == '__main__':
    main()

#------------------------------
