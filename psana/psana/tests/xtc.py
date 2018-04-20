import sys
from sys import getrefcount as getref
import numpy as np
from psana import DataSource

def myroutine2():
  ds = DataSource('data.xtc')  
  assert getref(ds)==2
  for evt in ds.events():
      dgram = evt.next()
      break
  assert getref(dgram)==7

  arr1 = dgram.xpphsd.raw.array0Pgp
  dgramObj=arr1.base
  assert getref(arr1)==3
  assert getref(dgramObj)==8
  s1 = arr1[2:4]
  assert s1.base is arr1
  assert getref(arr1)==4
  assert getref(s1)==2

  arr2 = dgram.xpphsd.raw.array0Pgp
  assert getref(dgramObj)==8
  s2 = arr2[3:5]
  assert s2.base is arr2
  assert getref(dgramObj)==8
  assert getref(arr2)==6

  arr3 = dgram.xppcspad.raw.arrayRaw
  assert getref(dgramObj)==8
  assert(arr3[7]==7)
  assert getref(dgramObj)==8
  assert getref(arr3)==3

  return s1, dgram, ds.dm.configs[0]

def myroutine1():
  s1, dgram, config =  myroutine2()
  assert getref(s1)==2

  return dgram, config

def myiter(dgram,testvals):
  for attrname,attr in dgram.__dict__.items():
    if hasattr(attr,'__dict__'):
      myiter(attr,testvals)
    else:
      if type(attr) is np.ndarray:
        assert np.array_equal(attr,testvals[attrname])
      elif type(attr) is not str and type(attr) is not tuple:
        assert attr==testvals[attrname]

def xtc():
  from .vals import testvals

  dgram, config = myroutine1()
  myiter(config,testvals)
  print('config tested',len(testvals),'values')
  myiter(dgram,testvals)
  print('xtc tested',len(testvals),'values')
