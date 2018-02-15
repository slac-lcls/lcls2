#!@PYTHON@
"""
Class :py:class:`GUViewColorBar` is a QWidget for interactive color bar
=======================================================================

Usage ::

    import sys
    from PyQt4 import QtGui, QtCore
    from graphqt.GUViewColorBar import GUViewColorBar
    import graphqt.ColorTable as ct
    app = QtGui.QApplication(sys.argv)
    ctab = ct.color_table_monochr256()
    w = GUViewColorBar(None, coltab=ctab)
    w.show()
    app.exec_()

Created on September 9, 2016 by Mikhail Dubrovin
"""

#-----------------------------

import numpy as np
from PyQt4 import QtGui, QtCore
import graphqt.ColorTable as ct
from graphqt.GUViewImage import GUViewImage, image_with_random_peaks
from pyimgalgos.NDArrGenerators import aranged_array

#-----------------------------

class GUViewColorBar(GUViewImage) :
    
    def __init__(self, parent=None, coltab=None, rulers='R', margl=None, margr=None, margt=None, margb=None) :
        #arr = image_with_random_peaks((1000, 1000))
        arr = aranged_array(shape=(5,500), dtype=np.uint32)

        w, h = arr.shape
        rectax = QtCore.QRectF(0, 0, w, h)
        origin = 'UL'
        scale_ctl = '' # 'HV'
        GUViewImage.__init__(self, parent, arr, coltab, origin, scale_ctl, rulers, margl, margr, margt, margb)

        #self.setMinimumSize(100,500)
        self.setFixedSize(100,500)

#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------
#-----------------------------

def test_guiviewcolorbar(tname) :
    print '%s:' % sys._getframe().f_code.co_name
    #import numpy as np
    #arr = np.random.random((1000, 1000))
    #arr = image_with_random_peaks((1000, 1000))
    #ctab = ct.color_table_monochr256()
    ctab = ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20)

    app = QtGui.QApplication(sys.argv)
    w = None
    if tname == '0': 
        w = GUViewColorBar(None, coltab=ctab, rulers='R', margl=0, margr=0.5, margt=0.03, margb=0.03)
        
    else :
        print 'test %s is not implemented' % tname
        return
    w.show()
    app.exec_()

#------------------------------

if __name__ == "__main__" :
    import sys; global sys
    #import numpy as np; global np
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print 50*'_', '\nTest %s' % tname
    test_guiviewcolorbar(tname)
    sys.exit('End of Test %s' % tname)

#------------------------------
