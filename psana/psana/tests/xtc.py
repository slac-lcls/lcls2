import sys
from sys import getrefcount as getref
import numpy as np
from psana import DataSource

def myroutine():
  ds = DataSource('data.xtc')  
  for nevent,evt in enumerate(ds.events()):
      if nevent==0:
        dgrambytes_event0=evt.dgrams[0]._dgrambytes
      elif nevent==1:
        dgram_event1 = evt.dgrams[0]

  # be sure you know what you are doing before you change
  # these reference count numbers. - cpo
  dgrambytes_event1 = dgram_event1._dgrambytes
  # 4 for arrays, 1 for dgram, 1 for getref, 1 for dgrambytes_event1
  assert getref(dgrambytes_event1)==7
  # event0 dgram is deleted, so only 1 for dgrambytes_event0 and 1 for getref
  assert getref(dgrambytes_event0)==2

  return dgram_event1, ds._configs()[0]

class DgramTester:
  def __init__(self,testvals):
    self.ntested=0
    self.testvals=testvals
  def iter(self,parent):
    for attrname,attr in parent.__dict__.items():
      if hasattr(attr,'__dict__'):
        self.iter(attr)
      else:
        if type(attr) is np.ndarray:
          assert np.array_equal(attr,self.testvals[attrname])
          self.ntested+=1
        elif type(attr) is not str and type(attr) is not tuple:
          assert attr==self.testvals[attrname]
          self.ntested+=1
    return self.ntested

def xtc():
  from .vals import testvals

  dgram, config = myroutine()
  configtester = DgramTester(testvals)
  ntested = configtester.iter(config)
  assert(ntested==len(testvals))
  dgtester = DgramTester(testvals)
  ntested = dgtester.iter(dgram)
  assert(ntested==len(testvals))
