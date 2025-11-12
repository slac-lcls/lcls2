#!/usr/bin/env python
"""
   class ArrayIterator: splits 3-D array for equal parts for last 2 indices (panels)

   import psana.detector.ArrayIteratorsplits as ai
   OR:
   from psana.detector.ArrayIterator import ArrayIterator, test_ArrayIterator
"""


class ArrayIterator:
    def __init__(self, shape, rstride=176, cstride=192): # cstride=48
        """ e.g. shape=(4, 352, 384), and rstride, cstride should split segment for equal parts"""
        assert(isinstance(shape, tuple))
        assert(len(shape)==3)
        self.shape = shape
        self.rstride = rstride
        self.cstride = cstride

    def __iter__(self):
        self.iseg = 0
        self.irow = 0
        self.icol =-self.cstride
        return self

    def __next__(self):
        """ returns tuple of slices with indices"""
        self.icol += self.cstride
        if self.icol == self.shape[2]:
           self.icol = 0
           self.irow += self.rstride
           if self.irow == self.shape[1]:
              self.irow = 0
              self.iseg += 1
              if self.iseg == self.shape[0]:
                 self.iseg = 0
                 raise StopIteration
        return (self.iseg, slice(self.irow, self.irow+self.rstride, 1), slice(self.icol, self.icol+self.cstride, 1))
        #return self.iseg, self.irow, self.icol


if __name__ == "__main__":

  import numpy as np
  import psana.detector.UtilsGraphics as ug
  gr = ug.gr

  def test_ArrayIterator(*args, **kwa):

    def image(nda):
        sh = nda.shape
        a = nda.copy()
        a.shape = (sh[0]*sh[1], sh[2])
        return a.T
        #print(ue.info_ndarr(a, 'img ', first=1000, last=1005))

    sh = (4, 352, 384)
    nda = np.ones(sh, dtype=np.int8)

    flimg = ug.fleximage(image(nda), arr=nda, h_in=5, w_in=15)

    for i,s in enumerate(ArrayIterator(sh, rstride=176, cstride=48)):  # cstride=192
        nda[s] = i
        print('%04d %s' % (i, str(s)))
        flimg.update(image(nda), arr=nda)
        gr.set_win_title(flimg.fig, titwin='region %d' % i)
        gr.show(mode='DO NOT HOLD')
    gr.show()

  print("begin test_ArrayIterator")
  test_ArrayIterator()

# EOF
