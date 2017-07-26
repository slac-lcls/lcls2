import sys
from sys import getrefcount as getref
sys.path.append('build/pdsdata')
import testType
def myroutine2():
  d = testType.Noddy()
  assert getref(d)==2
  arr1 = d.create() #increase refcount of d
  assert getref(d)==3
  assert getref(arr1)==2
  s1 = arr1[2:4] #increase refcount of arr1
  assert s1.base is arr1
  assert getref(arr1)==3
  assert getref(s1)==2

  arr2 = d.create() #increase refcount of d
  assert getref(d)==4
  s2 = arr2[3:5] # increase refcount of arr2
  assert getref(d)==4
  assert getref(arr2)==3

  return s1,d # destroys arr2 and s2, should decrement d refcnt by 1

def myroutine1():
  s1,d =  myroutine2()
  assert getref(d)==3
  assert getref(s1)==2

  return d # should decrement d refcnt by 1 since s1 is now gone

d = myroutine1()
assert getref(d)==2
