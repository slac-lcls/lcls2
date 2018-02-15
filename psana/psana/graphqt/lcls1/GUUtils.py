#!@PYTHON@
"""
:py:class:`GUUtils` - re-usable utils
======================================

Usage::

    from graphqt.GUUtils import select_item_from__popup_menu, QtGui
    app = QtGui.QApplication(sys.argv)    
    lst = ('apple', 'orange', 'pear', 'peache', 'plum') 
    item = select_item_in_popup_menu(lst)
    print 'selected: %s' % item

    from graphqt.GUUtils import print_rect
    print_rect(r, cmt='')

Created on 2016-10-21 by Mikhail Dubrovin
"""
#------------------------------

#import os
#from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt
#from PyQt4.QtGui import QGraphicsRectItem
from PyQt4 import QtGui

import numpy as np
import math

#------------------------------

#class GUUtils() :
#    def __init__(self) :
#        pass

#------------------------------

def print_rect(r, cmt='') :
    x, y, w, h = r.x(), r.y(), r.width(), r.height()
    L, R, T, B = r.left(), r.right(), r.top(), r.bottom()
    print '%s x=%8.2f  y=%8.2f  w=%8.2f  h=%8.2f' % (cmt, x, y, w, h)
    print '%s L=%8.2f  B=%8.2f  R=%8.2f  T=%8.2f' % (len(cmt)*' ', L, B, R, T)

#------------------------------

def select_item_from_popup_menu(list):
    """Shows the list as a pop-up menu and returns the selected item as a string or None"""
    w = QtGui.QMenu()
    for item in list : w.addAction(item)
    item = w.exec_(QtGui.QCursor.pos())
    return None if item is None else str(item.text()) # str(QString)

#------------------------------

def select_color(colini=Qt.blue, parent=None):
    """Select color using QColorDialog"""
    QCD = QtGui.QColorDialog
    w = QCD(colini, parent)
    w.setOptions(QCD.ShowAlphaChannel)# | QCD.DontUseNativeDialog | QCD.NoButtons
    res = w.exec_()
    color=w.selectedColor()
    #color = QtGui.QColorDialog.getColor()
    return None if color is None else color # QColor or None

#------------------------------

def proc_stat(weights, bins) :
    """ Copied from CalibManager/src/PlotImgSpeWidget.py
    """
    center = np.array([0.5*(bins[i] + bins[i+1]) for i,w in enumerate(weights)])

    sum_w  = weights.sum()
    if sum_w <= 0 : return  0, 0, 0, 0, 0, 0, 0, 0, 0
    
    sum_w2 = (weights*weights).sum()
    neff   = sum_w*sum_w/sum_w2 if sum_w2>0 else 0
    sum_1  = (weights*center).sum()
    mean = sum_1/sum_w
    d      = center - mean
    d2     = d * d
    wd2    = weights*d2
    m2     = (wd2)   .sum() / sum_w
    m3     = (wd2*d) .sum() / sum_w
    m4     = (wd2*d2).sum() / sum_w

    #sum_2  = (weights*center*center).sum()
    #err2 = sum_2/sum_w - mean*mean
    #err  = math.sqrt(err2)

    rms  = math.sqrt(m2) if m2>0 else 0
    rms2 = m2
    
    err_mean = rms/math.sqrt(neff)
    err_rms  = err_mean/math.sqrt(2)    

    skew, kurt, var_4 = 0, 0, 0

    if rms>0 and rms2>0 :
        skew  = m3/(rms2 * rms) 
        kurt  = m4/(rms2 * rms2) - 3
        var_4 = (m4 - rms2*rms2*(neff-3)/(neff-1))/neff if neff>1 else 0
    err_err = math.sqrt(math.sqrt(var_4)) if var_4>0 else 0 
    #print  'mean:%f, rms:%f, err_mean:%f, err_rms:%f, neff:%f' % (mean, rms, err_mean, err_rms, neff)
    #print  'skew:%f, kurt:%f, err_err:%f' % (skew, kurt, err_err)
    return mean, rms, err_mean, err_rms, neff, skew, kurt, err_err, sum_w

#------------------------------

def proc_stat_v2(weights, centers) :
    """ Returns mean and rms for input histogram arrays.

        Simplified version of proc_stat from CalibManager.PlotImgSpeWidget
        mean, rms, err_mean, err_rms, neff, skew, kurt, err_err, sum_w = proc_stat(weights,bins)
    """
    import math

    sum_w  = weights.sum()
    if sum_w == 0 : return  0, 0
    
    sum_w2 = (weights*weights).sum()
    neff   = sum_w*sum_w/sum_w2
    sum_1  = (weights*centers).sum()
    mean = sum_1/sum_w
    d      = centers - mean
    d2     = d * d
    wd2    = weights*d2
    m2     = (wd2)   .sum() / sum_w
    #m3     = (wd2*d) .sum() / sum_w
    #m4     = (wd2*d2).sum() / sum_w

    #sum_2  = (weights*center*center).sum()
    #err2 = sum_2/sum_w - mean*mean
    #err  = math.sqrt(err2)

    rms  = math.sqrt(m2) if m2>=0 else 0
    return mean, rms

#------------------------------

def equal_rects(r1, r2):
    if r1.left()   != r1.left() :   return False
    if r1.right()  != r1.right()  : return False
    if r1.bottom() != r1.bottom() : return False
    if r1.top()    != r1.top()    : return False
    return True

#------------------------------
#------------------------------
#------------------------------

def test_select_item_from_popup_menu():
    app = QtGui.QApplication(sys.argv)    
    lst = ('apple', 'orange', 'pear', 'peache', 'plum') 
    item = select_item_from_popup_menu(lst)
    print '%s is selected' % item

#------------------------------

def test_select_color():
    app = QtGui.QApplication(sys.argv)    
    color = select_color()
    print 'Selected color:', color

#------------------------------

if __name__ == "__main__" :
    import sys
    if len(sys.argv)<2 :
        test_select_item_from_popup_menu()
        sys.exit('Use command:\n> %s <test-id-string>' % sys.argv[0].split('/')[-1])
    print 'Test: %s' % sys.argv[1]
    if   sys.argv[1]=='1' : test_select_item_from_popup_menu()
    elif sys.argv[1]=='2' : test_select_color()
    elif sys.argv[1]=='3' : test01()
    else : sys.exit('Unknown test id: %s'%sys.argv[1])
    sys.exit('Test %s is completed' % sys.argv[1])

#------------------------------
