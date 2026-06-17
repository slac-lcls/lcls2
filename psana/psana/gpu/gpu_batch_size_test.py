"""Quick test: optimal_kernel_batch_size on A100."""
from psana.gpu import optimal_kernel_batch_size
import numpy as np
import cupy as cp

attrs   = cp.cuda.Device(0).attributes
n_sms   = attrs["MultiProcessorCount"]
tpb     = 256
bps     = min(attrs["MaxBlocksPerMultiprocessor"],
              attrs["MaxThreadsPerMultiProcessor"] // tpb)
cap     = n_sms * bps
print(f"A100: {n_sms} SMs x {bps} blocks/SM = {cap} concurrent blocks\n")

rows = [
    ("Jungfrau 4M GPU segs (19 segs)",  (19, 512, 1024)),
    ("Jungfrau 4M full   (32 segs)",    (32, 512, 1024)),
    ("Jungfrau 0.5M       (1 seg)",     (1,  512, 1024)),
    ("ePix100a            (2 segs)",    (2,  184, 388)),
    ("ePix HR             (2 segs)",    (2,  800, 768)),
    ("CSPAD full         (32 segs)",    (32, 185, 388)),
    ("Rayonix             (1 seg)",     (1, 1920, 1920)),
]

hdr = f"{'Detector':<38} {'pixels':>10} {'blks/evt':>10} {'opt_batch':>10} {'util%':>8}"
print(hdr)
print("-" * len(hdr))
for name, shape in rows:
    n_pix = int(np.prod(shape))
    bpe   = (n_pix + tpb - 1) // tpb
    opt   = optimal_kernel_batch_size(shape)
    util  = min(100, round(100 * opt * bpe / cap))
    print(f"{name:<38} {n_pix:>10,} {bpe:>10,} {opt:>10} {util:>7}%")

print("\n--- pipeline with batch_size=0 (auto) ---")
import glob
from psana import DataSource
smd = sorted(glob.glob(
    "/sdf/data/lcls/ds/prj/public01/xtc/smalldata/mfx100852324-r0077*"))
smd = list(dict.fromkeys(smd))
ds  = DataSource(files=smd, gpu_det="jungfrau", batch_size=0, max_events=3)
n   = 0
for run in ds.runs():
    for ctx in run.events():
        c = ctx.get("calib").on_gpu
        print(f"  event {n}: shape={c.shape} mean={float(c.mean()):.3f}")
        n += 1
print(f"Passed ({n} events with auto batch_size)")
