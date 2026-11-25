"""Time full 500k header index and 1003 bin-ordered sampled CPU references.

Diagnostic only: no numerical acceptance claim and no direct full-run estimate.
Outer launcher enforces a 600-second whole-process timeout.
"""
import json
import os
from pathlib import Path
import sys
import time
from support import HERE, PACKAGE, require, write

def main(inputs, output):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['PS_PARALLEL']='none'
    sys.path.insert(1, str(HERE/'vendor/sim10_compute_cube_20261115'))
    import reference
    reference.SNAPSHOT=PACKAGE/'calibration'
    started=time.monotonic()
    info=json.loads(inputs.read_text())
    constants, records=reference.load_constants()
    constants_seconds=time.monotonic()-started
    begin=time.monotonic()
    reader=reference.SeedReader(info['files'], info['timestamps'])
    index_seconds=time.monotonic()-begin
    selected=reference.samples()
    calibrate=reference.CpuCalibration(constants, selected)
    begin=time.monotonic()
    hashes=[]
    try:
        for bin_index in range(99):
            for ordinal in range(bin_index,1003,99):
                hashes.append(reference.digest(calibrate(reader.raw(info['timestamps'][ordinal], selected))))
    finally:
        reader.close()
    require(len(hashes)==1003, 'Sampled diagnostic coverage')
    write(output, dict(status='pass', full_index_events=100000, full_index_files=5,
                       index_seconds=index_seconds, sample_events=1003, sample_seconds=time.monotonic()-begin,
                       constants_seconds=constants_seconds, constants=records, hashes=hashes,
                       scope='Reference-cost diagnostic only. No full-reference completion or linear extrapolation claim.'))

if __name__=='__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
