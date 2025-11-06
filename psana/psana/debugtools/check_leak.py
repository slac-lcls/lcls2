# leak_probe_single.py
import os, time
import psutil
from psana import DataSource
import gc
import weakref
from collections import Counter

SUSPECT_PREFIXES = [
    # core psana object graphs
    "DataSource", "MPIDataSource", "NullDataSource",
    "Run", "RunParallel", "RunShmem", "RunSinglefile", "RunSerial",
    "Detector", "DetectorImpl", "SmallData",
    # readers/managers & infra
    "SmdReader", "SmdReaderManager", "DgramManager",
    "PrometheusManager", "ClientSocket", "Kafka",
    # calibration / geometry
    "Calib", "WeakDict", "WeakList", "SegGeometry", "Geometry",
    # specific dets you used
    "Jungfrau", "Archon",
    # common Python bits that often hold refs
    "Thread", "Queue", "Logger", "MappingProxyType",
    # big payloads (name only; not always cycles but memory-heavy)
    "ndarray",
]

def garbage_by_prefixes(prefixes, limit=None):
    """Yield objects in gc.garbage whose class name starts with any prefix."""
    found = 0
    for o in gc.garbage:
        if isinstance(o, (weakref.ProxyType, weakref.CallableProxyType, weakref.ReferenceType)):
            continue
        try:
            name = o.__class__.__name__
        except ReferenceError:
            continue
        if any(name.startswith(p) for p in prefixes):
            yield o
            found += 1
            if limit and found >= limit:
                return

def garbage_type_hist(n=30):
    c = Counter()
    for o in gc.garbage:
        try:
            c[o.__class__.__name__] += 1
        except ReferenceError:
            pass
    return c.most_common(n)

def print_chain(o, depth=0, seen=set()):
    if depth > 4 or id(o) in seen: return
    seen.add(id(o))
    print("  " * depth + f"{type(o).__name__}")
    for r in gc.get_referrers(o):
        # skip frames/modules to cut noise
        if any(t in str(type(r)) for t in ("frame", "module")): continue
        print_chain(r, depth+1, seen)

import ctypes
libc = ctypes.CDLL("libc.so.6")
def trim():
    try: libc.malloc_trim(0)
    except Exception: pass

EXP = os.environ.get("EXP", "mfx100848724")      # set to your Jungfrau exp
RUNS = [int(x) for x in os.environ.get("RUNS","51").split(",")]
DET  = os.environ.get("DET","jungfrau")          # Jungfrau alias
NLOOPS = int(os.environ.get("NLOOPS","20"))
FORCE_GC = bool(int(os.environ.get("FORCE_GC","1")))
FORCE_TRIM = bool(int(os.environ.get("FORCE_TRIM","1")))
SIMULATE_LEAK = bool(int(os.environ.get("SIMULATE_LEAK","0")))  # 1 => keep refs

LEAK_BAG = []  # strong refs (to simulate leaks)

def rss():
    return psutil.Process().memory_info().rss / 1024 ** 2  # MB
base = rss()

print(f"Start RSS={base:.1f} MB  exp={EXP} det={DET} runs={RUNS} loops={NLOOPS}", flush=True)

for i in range(NLOOPS):
    rn = RUNS[i % len(RUNS)]
    ds = DataSource(exp=EXP, run=rn, use_calib_cache=False)
    #ds = DataSource(files=['/sdf/data/lcls/ds/tmo/tmo101347825/xtc/tmo101347825-r0270-s000-c000.xtc2'])
    run = next(ds.runs())
    for evt in run.events():
        break  # get first event only

    #det = run.Detector(DET)  # forces calib load
    #evt = next(run.events())
    if DET == "jungfrau":
        img = det.raw.calib(evt)
        # Touch the constants so they materialize in memory:
        cc = det.raw._calibconst['pedestals']
        if cc:
            arr, meta = cc  # arr should be a numpy array (big)
            _ = getattr(arr, "nbytes", 0)
        if SIMULATE_LEAK and cc:
            LEAK_BAG.append(cc)  # <— this keeps a strong ref per run

    # Clean up Python names (what weakrefs *don’t* handle)
    #del det, run, ds

    if FORCE_GC:
        gc.set_debug(gc.DEBUG_SAVEALL)
        gc.collect()
        print("unreachable now parked in gc.garbage:", len(gc.garbage))

        # inspect a few objects, e.g. draw backrefs of one
        #print("Top garbage types:", garbage_type_hist(20))

        # Pull a few likely suspects
        suspects = list(garbage_by_prefixes(SUSPECT_PREFIXES, limit=5))
        print("Picked suspects:", [type(o).__name__ for o in suspects])

        # Peek at referrers (quick-n-dirty without objgraph)
        for i, o in enumerate(suspects):
            refs = [type(r).__name__ for r in gc.get_referrers(o)[:10]]
            print(f"[{i}] {type(o).__name__} referrers sample:", refs)
        sus = next(garbage_by_prefixes(["Run", "Detector", "SmdReaderManager",
                                "PrometheusManager", "SmallData", "DataSource", "MPIDataSource", "SerialDataSource"], limit=1), None)
        if sus: print_chain(sus)

        # IMPORTANT: free them; otherwise you’re pinning memory
        gc.garbage[:] = []
        gc.set_debug(0)
        gc.collect()
    if FORCE_TRIM:
        trim()

    print(f"Iter {i:02d} run {rn}: RSS Δ={rss()-base:.1f} MB  (bag={len(LEAK_BAG)})", flush=True)
    time.sleep(0.1)
