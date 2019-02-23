import sys
from sys import getrefcount as getref
import numpy as np
from psana import DataSource

def myroutine(fname):
  ds = DataSource(fname)  
  for nevent,evt in enumerate(ds.events()):
      if nevent==0:
        dgrambytes_event0=evt._dgrams[0]._dgrambytes
      elif nevent==1:
        dgram_event1 = evt._dgrams[0]

  # be sure you know what you are doing before you change
  # these reference count numbers. - cpo
  dgrambytes_event1 = dgram_event1._dgrambytes
  # 4 for arrays, 1 for dgram, 1 for getref, 1 for dgrambytes_event1
  assert getref(dgrambytes_event1)==7
  # event0 dgram is deleted, so only 1 for dgrambytes_event0 and 1 for getref
  #assert getref(dgrambytes_event0)==2

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
      # detector name is a dict with segment number as the key
      # test the values of the first segment
      if type(attr) is dict: attr=attr[0]
      if hasattr(attr,'__dict__'):
        self.depth+=1
        self.iter(attr)
        self.depth-=1
      else:
        if type(attr) is np.ndarray:
          assert np.array_equal(attr,self.testvals[attrname])
          self.ntested+=1
          print('*** test',attrname)
        elif type(attr) is not str and type(attr) is not tuple:
          assert attr==self.testvals[attrname]
          self.ntested+=1
          print('*** test',attrname)
    return self.ntested

def xtc(fname):

  import sys, os
  sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path
  from vals import testvals

  dgram, config = myroutine(fname)
  configtester = DgramTester(testvals)
  ntested = configtester.iter(config)
  assert(ntested==len(testvals))
  dgtester = DgramTester(testvals)
  ntested = dgtester.iter(dgram)
  assert(ntested==len(testvals))
