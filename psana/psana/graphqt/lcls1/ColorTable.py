#!@PYTHON@
"""
:py:class:`ColorTable` collection of methods to generate color bars
===================================================================

Usage ::

    import graphqt.ColorTable as ct

    ctab = ct.color_table_monochr256()
    ctab = ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20)
    ctab = ct.color_table_interpolated()
    ctab = ct.color_table_interpolated(points=[0, 100, 200, 400, 500, 650, 700], colors=[0xffffff, 0xffff00, 0x00ff00, 0xff0000, 0xff00ff, 0x0000ff, 0])

See:
    - :class:`IVMain`
    - :class:`IVMainTabs`
    - :class:`IVMainButtons`
    - :class:`IVImageCursorInfo`
    - :class:`IVConfigParameters`
    - :class:`IVTabDataControl`
    - :class:`IVTabFileName`
    - `graphqt documentation <https://lcls-psana.github.io/graphqt/py-modindex.html>`_.

Created on Dec 6, 2015 by Mikhail Dubrovin
"""
#------------------------------

from PyQt4 import QtGui #, QtCore
from PyQt4.QtCore import Qt
import numpy as np
from math import floor

#------------------------------

class Storage :
    """Store for shared parameters."""
    def __init__(self) :
        self.ictab = 0

    def color_table_index(self) :
        return self.ictab

#------------------------------
STOR = Storage()
#------------------------------

def print_colors(arr) :
    sh = arr.shape
    for row in arr :
      for v in row :
        qc = QtGui.QColor(v & 0xFFFFFF) # rgb part only
        #print v, qc.red(), qc.green(), qc.blue() 
        print '%4d' % qc.red(),
      print ''


def apply_color_table(arr, ctable=None, amin=None, amax=None) :
    ''' Returns numpy array with colors in stead of intensities
    '''
    # color_table_monochr256()
    # color_table_linear(ncolors=100)
    ctab = ctable if ctable is not None else color_table_rainbow(ncolors=1000, hang1=0, hang2=360)
    # color_table_monochr256() # color_table_def()
    min = np.amin(arr) if amin is None else amin
    max = np.amax(arr) if amax is None else amax
    if min==max : max+=1
    f = float(ctab.size-1)/(max-min)
    ict = np.require(f*(arr-min), dtype=np.int) # array of indexes in color table

    imax = len(ctab) - 1
    cond = np.logical_and(ict>0, ict<len(ctab))
    ict = np.select((cond, ict>imax), (ict, imax), default=0)

    return ctab[ict]

    #print 'XXX:indexes:\n', ict
    #a = ctab[ict]    
    #print 'XXX:arr of colors:\n', a
    #print_colors(a)
    #return a


def color_table_monochr256(inverted=False) :
    ''' Returns numpy array with monochrome table of 256 colors
    '''
    ncolors=256
    inds = range(ncolors-1,-1,-1) if inverted else range(ncolors)
    return np.array([c + c*0x100 + c*0x10000 + 0xff000000 for c in inds], dtype=np.uint32)  


def color_table_linear(ncolors=100) :
    ''' Returns numpy array with ncolors constructed from entire range of RGB colors
    '''
    f = 1./ncolors
    return np.array([0xffffff*c*f + 0xff000000 for c in range(ncolors)], dtype=np.uint32)  


def interpolate_colors(ctab, p1, p2, c1, c2) :
    '''Fills color table ctab between index/points p1 and p2 for interpolated colors from c1 to c2
    '''
    #print p1, p2, c1, c2
    A = 0xff000000
    R = 0x00ff0000
    G = 0x0000ff00
    B = 0x000000ff
    r1, r2 = (c1&R)>>16, (c2&R)>>16
    g1, g2 = (c1&G)>>8,  (c2&G)>>8
    b1, b2 = c1&B, c2&B
    np = p2-p1
    #print 'XXX: c1, c2, p1, p2, r1, r2, g1, g2,  b1, b2:', hex(c1), hex(c2), p1, p2, r1, r2, g1, g2, b1, b2

    if np<1 :
        ctab[p1] = c1 + A
        return

    fr = float(r2-r1) / np
    fg = float(g2-g1) / np
    fb = float(b2-b1) / np

    for p in range(p1,p2) :
        dp = p-p1

        r = r1 + int(floor(fr*dp))
        g = g1 + int(floor(fg*dp))
        b = b1 + int(floor(fb*dp))

        ctab[p] = A + b + g*0x100 + r*0x10000

        #print 'hex(r), hex(r<<16)', hex(r), hex(r<<16)
        #print 'hex(b), hex(b<<8)',  hex(b), hex(b<<8)

        #ctab[p] = A + (b + g<<8 + r<<16) & 0xffffff

        #color = ctab[p]
        #qc = QtGui.QColor(color & 0xFFFFFF) # rgb part only
        #print 'XXX: point:%4d   %10s %4d %4d %4d' % (p, hex(color), qc.red(), qc.green(), qc.blue())


#def color_table_interpolated(points=[0,      200,      400,      600,      800],\
#                             colors=[0, 0x0000ff, 0x00ff00, 0xff0000, 0xffffff]) :
#def color_table_interpolated(points=[0,      200,      400,      600,      800],\
#                             colors=[0, 0x0000ff, 0xff0000, 0x00ff00, 0xffffff]) :
def color_table_interpolated(points=[0,      50,      200,      300,      500,      600,      700],\
                             colors=[0, 0x0000ff, 0xff00ff, 0xff0000, 0x00ff00, 0xffff00, 0xffffff]) :
    ''' Returns numpy array of colors linearly-interpolated between points with defined colors
    '''
    #print 'XXX: number of colors: %d' % points[-1]
    ctab = np.zeros(points[-1], dtype=np.uint32)
    for i,p in enumerate(points[:-1]) :
        p1, p2 = p, points[i+1]
        c1, c2 = colors[i], colors[i+1]
        interpolate_colors(ctab, p1, p2, c1, c2)
    return ctab
    #return color_table_monochr256()


def color_table_rainbow(ncolors=1000, hang1=250, hang2=-20) :
    ct = ColorTable(ncolors, hang1, hang2)
    return ct.np_ctable()

#------------------------------

def next_color_table(ict=None) :
    """Returns color table selected in loop or requested by index ict : int among pre-defined
    """
    if ict is None : STOR.ictab += 1
    else           : STOR.ictab = ict
    #print 'Color table # %d' % STOR.ictab
    if   STOR.ictab == 2 : return color_table_rainbow(ncolors=1000, hang1=-20, hang2=250)
    elif STOR.ictab == 3 : return color_table_monochr256()
    elif STOR.ictab == 4 : return color_table_monochr256(inverted=True)
    elif STOR.ictab == 5 : return color_table_rainbow(ncolors=1000, hang1=-120, hang2=100)
    elif STOR.ictab == 6 : return color_table_rainbow(ncolors=1000, hang1=100, hang2=-120)
    elif STOR.ictab == 7 : return color_table_interpolated()
    elif STOR.ictab == 8 : return color_table_interpolated(points=[0, 100, 200, 400, 500, 650, 700],\
                                colors=[0xffffff, 0xffff00, 0x00ff00, 0xff0000, 0xff00ff, 0x0000ff, 0]) 
    else :
        STOR.ictab = 1
        return color_table_rainbow()
        #return color_table_monochr256()
        #return color_table_interpolated()

#------------------------------

def get_pixmap(ind, orient='H', size=(200,30)) :
    ctab = next_color_table(ict=ind)
    arr  = array_for_color_bar(ctab, orient=orient)#, width = 10)
    (h,w) = arr.shape
    image = QtGui.QImage(arr, w, h, QtGui.QImage.Format_ARGB32)
    pixmap= QtGui.QPixmap.fromImage(image.scaled(size[0], size[1], Qt.IgnoreAspectRatio, Qt.FastTransformation))
    return pixmap

#------------------------------

def array_for_color_bar(ctab=color_table_monochr256(), orient='V', width=2) : 
    """Returns 2-d array made of repeated 1-d array ctab to display as a color bar
    """
    arr = [(c,c) for c in ctab[::-1]] if orient=='V' else\
          [ctab for r in range(width)]
    npa = np.array(arr, dtype=np.uint32) #.T
    #print 'XXX array for color bar:\n', npa
    #print 'XXX shape: ', npa.shape
    return npa

#------------------------------
#------------------------------
#------------------------------

class ColorTable():
    '''Creates and provide access to color table
    '''
    def __init__(self, ncolors=1000, hang1=0, hang2=360, vmin=-10000, vmax=10000):
        '''Makes color table - list of QColors of length ncolors 
        '''
        self.make_ctable_for_hue_range(ncolors, hang1, hang2)
        self.set_value_range(vmin, vmax)

    
    def make_ctable_for_hue_range(self, ncolors=1000, hang1=0, hang2=360) :
        '''Makes color table in the range of hue values
        '''
        self.ncolors = ncolors
        self.hang1 = float(hang1)
        self.hang2 = float(hang2)

        dhang = (self.hang2 - self.hang1)/ncolors
        self.ctable = []
        for ic in range(ncolors) :
            hang = self.hang1 + dhang * ic
            hnorm = float(hang % 360)/360.
            #print 'hang:%7.2f  rest(hang,360)=%7.2f' % (hang, hang % 360)
            qc = QtGui.QColor()
            qc.setHsvF(hnorm, 1., 1., alpha=1.)
            self.ctable.append(qc)


    def int_ctable(self) :
        '''converts list of QColor to list of integer rgba values
        '''
        return [c.rgba() for c in self.ctable]


    def np_ctable(self) :
        return np.array(self.int_ctable(), dtype=np.uint32)


    def set_ncolors(self, ncolors) :
        '''Sets the number of color in table and re-generate color table
        '''
        self.make_ctable_for_hue_range(ncolors, self.hang1, self.hang2)
        self.set_value_range(self.vmin, self.vmax)

        
    def set_hue_range(self, hang1, hang2) :
        '''Sets the range of hue angles and re-generate color table
        '''
        self.make_ctable_for_hue_range(self.ncolors, hang1, hang2)
        self.set_value_range(self.vmin, self.vmax)

                
    def set_value_range(self, vmin, vmax) :
        '''Sets the range of values which will be mapped to color table
        '''
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.vfct = self.ncolors/(self.vmax - self.vmin)


    def color_for_value(self, v) :
        '''Returns color mapped to the value in the color table
        '''
        if   v < self.vmin : return self.ctable[0]
        elif v < self.vmax : return self.ctable[self.vfct*v]
        else               : return self.ctable[-1]

      
    def print_color_table(self):    
        for ic, qc in enumerate(self.ctable) :
            print 'i:%4d  R:%3d  G:%3d  B:%3d' % (ic, qc.red(), qc.green(), qc.blue())

#------------------------------

def test():
    ct = ColorTable()
    ct.print_color_table()
    
#------------------------------

if __name__ == '__main__':
    test()

#------------------------------       
