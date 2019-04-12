import sys
from sys import getrefcount as getref
import numpy as np
from psana import DataSource

def refcnt_test(fname,nsegments,cydgram):
  ds = DataSource(fname)  
  for nevent,evt in enumerate(ds.events()):
      if nevent == 0:
        dgrambytes_event0 = evt._dgrams[0]._dgrambytes
      elif nevent == 1:
        dgram_event1 = evt._dgrams[0]

  # be sure you know what you are doing before you change
  # these reference count numbers. - cpo
  dgrambytes_event1 = dgram_event1._dgrambytes
  # 4 arrays per segment, 1 for dgram, 1 for getref, 1 for dgrambytes_event1
  # cydgram test eliminates 3 of these arrays however (a hack since
  # the xpphsd detector has unsupported cydgram types of charstr/enum)
  if cydgram:
    assert getref(dgrambytes_event1) == 1*nsegments+3
  else:
    assert getref(dgrambytes_event1) == 4*nsegments+3

  # event0 dgram is deleted, so only 1 for dgrambytes_event0 and 1 for getref
  assert getref(dgrambytes_event0) == 2

  return dgram_event1, ds._configs[0]

from vals import testvals

def valtester(dg, nsegments, cydgram):
  for iseg in range(nsegments):

    if not cydgram:
      hsdfex = dg.xpphsd[iseg].fex
      hsdraw = dg.xpphsd[iseg].raw
      assert np.array_equal(hsdraw.array0Pgp,testvals['array0Pgp'])
      assert np.array_equal(hsdraw.array1Pgp,testvals['array1Pgp'])
      assert hsdraw.floatPgp == testvals['floatPgp']
      assert hsdraw.intPgp == testvals['intPgp']

      assert hsdfex.floatFex == testvals['floatFex']
      assert np.array_equal(hsdfex.arrayFex,testvals['arrayFex'])
      assert hsdfex.intFex == testvals['intFex']

      assert hsdfex.charStrFex == testvals['charStrFex']
      assert hsdfex.enumFex1.value == testvals['enumFex1']['value']
      assert hsdfex.enumFex1.names == testvals['enumFex1']['names']
      assert hsdfex.enumFex2.value == testvals['enumFex2']['value']
      assert hsdfex.enumFex2.names == testvals['enumFex2']['names']
      assert hsdfex.enumFex3.value == testvals['enumFex3']['value']
      assert hsdfex.enumFex3.names == testvals['enumFex3']['names']

    cspadraw = dg.xppcspad[iseg].raw
    assert np.array_equal(cspadraw.arrayRaw,testvals['arrayRaw'])

def xtc(fname, nsegments, cydgram=False):

  import sys, os
  sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path

  dgram, config = refcnt_test(fname,nsegments,cydgram)
  valtester(config, nsegments, cydgram)
  valtester(dgram, nsegments, cydgram)
