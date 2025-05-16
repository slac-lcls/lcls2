import sys

import numpy as np
import vals

from psana import DataSource

known_epics_pedestals = np.array([[11.,12.,13.,14.,15.,16.],
     [21.,22.,23.,24.,25.,26.],
     [31.,32.,33.,34.,35.,36.]])

def launch_client(pid, supervisor=-1, supervisor_ip_addr=None, cached_detectors=None):
    dg_count = 0

    use_cache = cached_detectors is not None and "jungfrau" in cached_detectors

    kwargs = {
        "shmem": f"shmem_test_{pid}",
        "log_level": "DEBUG"
    }

    if use_cache:
        kwargs["use_calib_cache"] = True
        kwargs["cached_detectors"] = cached_detectors

    if supervisor != -1:
        kwargs["supervisor"] = supervisor
        kwargs["supervisor_ip_addr"] = supervisor_ip_addr

    ds = DataSource(**kwargs)
    run = next(ds.runs())

    if use_cache:
        # Detector cache test path — do not assert, just access jungfrau
        jungfrau = run.Detector("jungfrau")
        for evt in run.events():
            img = jungfrau.raw.image(evt)
            print(f'Received {dg_count=} {img.shape=}')
            dg_count += 1
    else:
        # Default path for existing shared-memory regression tests
        assert np.array_equal(ds.dsparms.calibconst['epics']['pedestals'][0], known_epics_pedestals)

        cspad = run.Detector('xppcspad')
        hsd = run.Detector('xpphsd')
        for evt in run.events():
            assert(hsd.raw.calib(evt).shape == (5,))
            assert(hsd.fex.calib(evt).shape == (6,))
            padarray = vals.padarray
            assert np.array_equal(cspad.raw.calib(evt), np.stack((padarray, padarray)))
            assert np.array_equal(cspad.raw.image(evt), np.vstack((padarray, padarray)))
            dg_count += 1

    return dg_count

# ------------------------------

def main():
    args = sys.argv
    use_cache = '--test-detector-cache' in args
    cached_detectors = ["jungfrau"] if use_cache else None

    # Remove custom flag before normal parsing
    args = [arg for arg in args if arg != '--test-detector-cache']

    if len(args) == 2:
        sys.exit(launch_client(args[1], cached_detectors=cached_detectors))
    elif len(args) == 4:
        sys.exit(launch_client(args[1], supervisor=int(args[2]), supervisor_ip_addr=args[3], cached_detectors=cached_detectors))

# ------------------------------

if __name__ == '__main__':
    main()
