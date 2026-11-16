"""Bounded CPU seed cache and per-bin CPU-v3 reference, independent of GPU sums."""
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np

from dense_cube import adapter, STAGE2
import gpu_accept as stage2
from fixture import file_sha, require
from psana import DataSource

SHAPE=(32,512,1024)
WORDS=52429
SEED_BYTES=32*WORDS*2


def prepare(root, directory):
    directory=Path(directory);directory.mkdir(mode=0o700,exist_ok=False)
    manifest=stage2.preflight(root)
    generation_sha=file_sha(root/'generation.json')
    baseline=json.loads((STAGE2/'validation-summary.json').read_text())
    witnesses=[v for v in baseline['fixtures'].values() if Path(v['generation']['path'])==root/'generation.json']
    require(len(witnesses)==1 and generation_sha==witnesses[0]['generation']['sha256'],'Unaccepted fixture')
    stamps=[r['timestamp'] for r in manifest['schedule'] if r['service']==12]
    adapter.register_reader()
    ds=DataSource(exp='mfx101555026',run=10,dir=str(root),live=False,skip_calib_load='all')
    run=next(ds.runs());reader=run.Detector('jungfrau').sim10
    packed=np.empty((32,WORDS),np.uint16)
    seen=[];start=time.monotonic()
    with (directory/'seeds.bin').open('xb') as out, closing(run.events()) as events:
        for evt in events:
            i=len(seen);require(i<len(stamps) and int(evt.timestamp)==stamps[i],'CPU seed ordering mismatch')
            seeds=reader.seed(evt);require(set(seeds)==set(range(32)),'Missing seed panel')
            for panel,seed in seeds.items():
                require(seed.shape==(WORDS,) and seed.dtype==np.uint16,'Seed layout mismatch')
                packed[panel]=seed
            packed.tofile(out);seen.append(int(evt.timestamp))
        out.flush();os.fsync(out.fileno())
    require(seen==stamps,'Incomplete CPU seed cache')
    report=dict(status='pass',fixture=str(root),generation_sha256=generation_sha,timestamps=stamps,
                seed_bytes_per_event=SEED_BYTES,seed_cache_sha256=file_sha(directory/'seeds.bin'),
                seed_cache_bytes=(directory/'seeds.bin').stat().st_size,elapsed_seconds=time.monotonic()-start)
    require(report['seed_cache_bytes']==len(stamps)*SEED_BYTES,'Cache size mismatch')
    (directory/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def verify(directory, cache, constants, timestamps, nbins, active_bins, retained, memory_sample):
    """Validate every saved pixel, holding only one bin's reference/bound at once."""
    cache=Path(cache);directory=Path(directory);retained=Path(retained)
    record=json.loads((cache/'manifest.json').read_text())
    require(record['timestamps']==timestamps and record['seed_cache_sha256']==file_sha(cache/'seeds.bin'),'Seed cache changed')
    exported=json.loads((directory/'manifest.json').read_text())
    expected_counts=np.bincount(np.arange(len(timestamps))%active_bins,minlength=nbins)
    require(np.array_equal(exported['counts'],expected_counts),'Wrong exported counts')
    source=DataSource(exp='mfx101555026',run=10,dir='/sdf/data/lcls/ds/prj/public01/xtc',max_events=1,live=False,skip_calib_load='all')
    source_run=next(source.runs());det=source_run.Detector('jungfrau',status=False);det.raw._calibconst=constants
    expected=np.empty(SHAPE,np.float64);absolute=np.empty(SHAPE,np.float64);raw=np.empty(SHAPE,np.uint16)
    rows=[];start=time.monotonic();retained.mkdir(exist_ok=False)
    representatives={0,active_bins-1,nbins-1}
    with (cache/'seeds.bin').open('rb') as f, closing(source_run.events()) as originals:
        original=next(originals)
        for b in range(nbins):
            expected.fill(0);absolute.fill(0)
            indices=range(b,len(timestamps),active_bins) if b<active_bins else ()
            count=0
            for i in indices:
                f.seek(i*SEED_BYTES);seed=np.fromfile(f,dtype=np.uint16,count=32*WORDS).reshape(32,WORDS)
                for panel in range(32):raw[panel]=stage2.expand(seed[panel])
                image=det.raw.calib(original,cversion=3,cmpars=None,nda_raw=raw).reshape(SHAPE)
                require(image.dtype==np.float32 and np.isfinite(image).all(),'Invalid CPU reference')
                expected+=image;absolute+=np.abs(image);count+=1
            require(count==int(expected_counts[b]),'Reference event count mismatch')
            path=directory/f'bin-{b:05d}.npy';actual=np.load(path,allow_pickle=False)
            require(actual.shape==SHAPE and actual.dtype==np.float64,'Export layout mismatch')
            eps=np.finfo(np.float32).eps/2;gamma=count*eps/(1-count*eps)
            # Compare panel by panel to keep error/bound scratch small.
            max_error=max_ratio=0.
            for panel in range(32):
                error=np.abs(actual[panel]-expected[panel]);bound=gamma*absolute[panel]+1e-5
                require(np.isfinite(actual[panel]).all() and np.all(error<=bound),'Sum outside float32 rounding bound')
                max_error=max(max_error,float(error.max()));max_ratio=max(max_ratio,float((error/bound).max()))
            if not count:require(not np.any(actual),'Nonzero empty bin')
            h=hashlib.sha256(memoryview(actual).cast('B')).hexdigest()
            rows.append(dict(bin=b,count=count,max_abs_error=max_error,max_bound_ratio=max_ratio,
                             pixel_sha256=h,file_sha256=file_sha(path),bytes=path.stat().st_size))
            if b in representatives:
                shutil.copyfile(path,retained/path.name)
                require(file_sha(retained/path.name)==rows[-1]['file_sha256'],'Retained copy differs')
            memory_sample('cpu_validation_bin')
            del actual
    return dict(status='pass',bins=rows,all_counts_exact=True,all_pixels_within_bound=True,
                compared_pixels=nbins*np.prod(SHAPE).item(),cpu_events_calibrated=int(expected_counts.sum()),
                elapsed_seconds=time.monotonic()-start,retained_bins=sorted(representatives),
                tolerance='abs(error) <= gamma_n*sum(abs(float32 input))+1e-5; u=2^-24; gamma_n=n*u/(1-n*u)',
                reference='CPU DataSource seed cache, np.repeat expansion, Jungfrau CPU cversion=3; independent of GPU output')
