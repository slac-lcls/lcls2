#!/usr/bin/env python
#

import sys
from sys import getrefcount as getref
sys.path.append('../../build/psana')
sys.path.append('../')
from dgram import Dgram
import numpy as np
from DataSource import DataSource
#from DataSourceContainer import DataSource

def myroutine2():
  ds = DataSource('data.xtc')
  assert getref(ds)==2
  d = ds.__next__()
  assert getref(d)==3
  arr1 = d.array0_pgp
  dgramObj=arr1.base
  assert getref(arr1)==3
  assert getref(dgramObj)==5
  s1 = arr1[2:4]
  assert s1.base is arr1
  assert getref(arr1)==4
  assert getref(s1)==2

  arr2 = d.array0_pgp
  assert getref(dgramObj)==5
  s2 = arr2[3:5]
  assert getref(dgramObj)==5
  assert getref(arr2)==6

  return s1,d

def myroutine1():
  s1,d =  myroutine2()
  assert getref(s1)==2

  return d

d = myroutine1()
from testvals import testvals
for key in testvals:
  val = testvals[key]
  if type(val) is np.ndarray:
    assert np.array_equal(val,getattr(d,key))
  else:
    assert val==getattr(d,key)
print('xtc tested',len(testvals),'values')
