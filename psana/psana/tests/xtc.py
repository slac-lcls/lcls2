import sys
from sys import getrefcount as getref
import numpy as np
from psana import DataSource

def myroutine2():
  ds = DataSource('data.xtc')  
  for evt in ds.events():
      break
  dgram = evt.dgrams[0]
  evt = None # to remove reference count that evt has to dgram to simplify test

  # be sure you know what you are doing before you change
  # these reference count numbers. - cpo
  assert getref(dgram)==2

  arr1 = dgram.xpphsd.raw.array0Pgp
  assert getref(dgram)==3

  assert arr1.base is dgram
  s1 = arr1[2:4]
  assert getref(arr1)==3
  assert s1.base is arr1
  assert getref(s1)==2

  arr2 = dgram.xpphsd.raw.array0Pgp
  assert getref(dgram)==4
  s2 = arr2[3:5]
  assert s2.base is arr2
  assert getref(dgram)==4

  arr3 = dgram.xppcspad.raw.arrayRaw
  assert getref(dgram)==5
  assert(arr3[7]==7)

  return s1, dgram, ds.dm.configs[0]

def myroutine1():
  s1, dgram, config =  myroutine2()
  assert getref(s1)==2
  assert getref(dgram)==3
  return dgram, config

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

  dgram, config = myroutine1()
  configtester = DgramTester(testvals)
  ntested = configtester.iter(config)
  assert(ntested==len(testvals))
  dgtester = DgramTester(testvals)
  ntested = dgtester.iter(dgram)
  assert(ntested==len(testvals))
