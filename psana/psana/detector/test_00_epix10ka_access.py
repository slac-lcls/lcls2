#!/usr/bin/env python

fname0 = '/reg/g/psdm/detector/data2_test/xtc/data-tstx00417-r0014-epix10kaquad-e000005.xtc2'
fname1 = '/reg/g/psdm/detector/data2_test/xtc/data-tstx00417-r0014-epix10kaquad-e000005-seg1and3.xtc2'

from psana import DataSource
ds = DataSource(files=fname0)
run = next(ds.runs())
det = run.Detector('epix10k2M')

print('XXX dir(det):\n', dir(det))
print('XXX dir(run):\n', dir(run))

raw = det.raw
print('dir(det.raw):', dir(det.raw))

print('raw._configs:', raw._configs)
cfg = raw._configs[0]
print('dir(cfg):', dir(cfg))

c0 = cfg.epix10k2M[0]
print('dir(c0):', dir(c0))

print('dir(c0.raw):', dir(c0.raw))

print('WHAT IS THAT? c0.raw.raw', c0.raw.raw)
