import sys
from sys import getrefcount as getref
import numpy as np
from psana import DataSource

def myroutine(fname,nsegments):
  ds = DataSource(fname)  
  for nevent,evt in enumerate(ds.events()):
      if nevent==0:
        dgrambytes_event0=evt._dgrams[0]._dgrambytes
      elif nevent==1:
        dgram_event1 = evt._dgrams[0]

  # be sure you know what you are doing before you change
  # these reference count numbers. - cpo
  dgrambytes_event1 = dgram_event1._dgrambytes
  # 4 arrays per segment, 1 for dgram, 1 for getref, 1 for dgrambytes_event1
  assert getref(dgrambytes_event1)==4*nsegments+3
  # event0 dgram is deleted, so only 1 for dgrambytes_event0 and 1 for getref
  assert getref(dgrambytes_event0)==2

  return dgram_event1, ds._configs[0]

class DgramTester:
  def __init__(self,testvals):
    self.ntested=0
    self.testvals=testvals
    self.depth=0
  def iter(self,parent):
    for attrname,attr in parent.__dict__.items():
      #print(' '*2*(self.depth),attrname)
      if attrname.startswith('_'): continue
      if type(attr) is dict:
        # detector name is a dict with segment number as the key
        for _,value in attr.items():
          self.depth+=1
          self.iter(value)
          self.depth-=1
      elif hasattr(attr,'__dict__'):
        self.depth+=1
        self.iter(attr)
        self.depth-=1
      else:
        if type(attr) is str:
          # ignore software/dettype/detid
          if attrname=='charStrFex':
            assert attr==self.testvals[attrname]
            self.ntested+=1
        elif type(attr) is np.ndarray:
          assert np.array_equal(attr,self.testvals[attrname])
          self.ntested+=1
        elif type(attr) is not tuple:
          assert attr==self.testvals[attrname]
          self.ntested+=1
    return self.ntested

def xtc(fname, nsegments, cydgram=False):

  import sys, os
  sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path
  from vals import testvals

  dgram, config = myroutine(fname,nsegments)
  configtester = DgramTester(testvals)
  ntested = configtester.iter(config)
  if cydgram:
    assert(ntested==(len(testvals)-1)*nsegments) # hack, since cydgram doesn't support CHARSTR
  else:
    assert(ntested==(len(testvals))*nsegments)
  dgtester = DgramTester(testvals)
  ntested = dgtester.iter(dgram)
  if cydgram:
    assert(ntested==(len(testvals)-1)*nsegments) # hack, since cydgram doesn't support CHARSTR
  else:
    assert(ntested==(len(testvals))*nsegments)
