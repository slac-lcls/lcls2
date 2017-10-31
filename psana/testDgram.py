#!/usr/bin/env python
#

import sys
from sys import getrefcount as getref
sys.path.append('../build/psana')
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
assert d.float_pgp==1.0
assert d.int_pgp==2
testarr = np.array([[0,0,0],[0,1,2],[0,2,4]],dtype=np.float32)
testarrB = testarr+2
assert np.array_equal(d.array0_pgp,testarr)
assert np.array_equal(d.array1_pgp,testarrB)
testfexarr = np.arange(142,148,dtype=np.float32).reshape([2,3])
assert np.array_equal(d.array_fex,testfexarr)
assert d.int_fex==42
assert d.float_fex==41.0
print('dgram test complete')
