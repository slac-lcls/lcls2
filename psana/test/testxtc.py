#!/usr/bin/env python
#

import sys
from sys import getrefcount as getref
sys.path.append('../../build/psana')
from dgram import Dgram
import numpy as np
sys.path.append('../')
from DataSource import DataSource
#from DataSourceContainer import DataSource

def myroutine2():
  ds = DataSource('data.xtc')
  assert getref(ds)==2
  e = ds.__next__()
  assert getref(e)==2
  arr1 = e.array0_pgp
  dgramObj=arr1.base
  assert getref(arr1)==3
  assert getref(dgramObj)==5
  s1 = arr1[2:4]
  assert s1.base is arr1
  assert getref(arr1)==4
  assert getref(s1)==2

  arr2 = e.array0_pgp
  assert getref(dgramObj)==5
  s2 = arr2[3:5]
  assert getref(dgramObj)==5
  assert getref(arr2)==6

  return s1,e

def myroutine1():
  s1,e =  myroutine2()
  assert getref(s1)==2

  return e

e = myroutine1()
from testvals import testvals
for key in testvals:
  val = testvals[key]
  if type(val) is np.ndarray:
    assert np.array_equal(val,getattr(e,key))
  else:
    assert val==getattr(e,key)
print('xtc tested',len(testvals),'values')
