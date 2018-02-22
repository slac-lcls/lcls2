#!@PYTHON@
"""
Class :py:class:`GUViewHist` is a QWidget for interactive image
===============================================================

Usage ::

    Create GUViewHist object within pyqt QApplication
    --------------------------------------------------
    import sys
    from PyQt4 import QtGui, QtCore
    from graphqt.GUViewHist import GUViewHist
    rectax=QtCore.QRectF(0, -1.2, 100, 2.4)    
    app = QtGui.QApplication(sys.argv)
    w = GUViewHist(None, rectax, origin='DL', scale_ctl='H', rulers='DL',\
                    margl=0.12, margr=0.02, margt=0.02, margb=0.06)
    w._ymin =-1.2
    w._ymax = 1.2
    w.add_hist(y1, (0,100), pen=QtGui.QPen(Qt.blue, 0), brush=QtGui.QBrush(Qt.green))
    w.show()
    app.exec_()

    Connect/disconnecr recipient to signals
    ---------------------------------------
    w.connect_statistics_updated_to(recip)
    w.disconnect_statistics_updated_from(recip)
    w.test_statistics_std_reception(self, mean, rms, err_mean, err_rms, neff, skew, kurt, err_err, sum_w)
 
    w.connect_mean_std_updated_to(recip)
    w.disconnect_mean_std_updated_from(recip)
    w.test_hist_mean_std_updated_reception(self, mean, rms)

    w.connect_cursor_bin_changed_to(recip)
    w.disconnect_cursor_bin_changed_from(recip)

    w.connect_histogram_updated_to(recip) :
    w.disconnect_histogram_updated_from(recip) :
    w.test_histogram_updated_reception()


    Methods
    -------
    w.set_limits_horizontal(amin=None, amax=None) # sets self.amin, self.amax
    w.set_limits()
    values, centers, edges = w.visible_hist_vce(ihis=0)
    mean, rms, err_mean, err_rms, neff, skew, kurt, err_err, sum_w = w.visible_hist_stat(ihis=0)
    mean, std = w.visible_hist_mean_std(ihis=0)

    w.on_axes_limits_changed(x1, x2, y1, y2)
    w.evaluate_hist_statistics() # emits signal statistics_updated
    w.evaluate_hist_mean_std() # emits signal mean_std_updated

    ind, hb.values = w.hist_bin_value(x, ihis=0)

    w.reset_original_hist() 
    w.add_hist(values, edges, pen=QtGui.QPen(Qt.black), brush=QtGui.QBrush(), vtype=np.float)
    w.add_array_as_hist(arr, pen=QtGui.QPen(Qt.black), brush=QtGui.QBrush(), vtype=np.float)
    w.remove_all_graphs()


    Internal methods
    -----------------
    w._init_cursor_scope()
    w.set_style()
    w._add_path_to_scene(path, pen=QtGui.QPen(Qt.yellow), brush=QtGui.QBrush())


    Re-defines methods
    ------------------
    mouseReleaseEvent, closeEvent, mouseMoveEvent, wheelEvent, 
    enterEvent, leaveEvent, keyPressEvent, __del__

    Global scope methods
    --------------------
    amin, amax, nhbins, values = image_to_hist_arr(arr, vmin=None, vmax=None, nbins=None)

Created on September 9, 2016 by Mikhail Dubrovin
"""

#import os
#import math
#import math
from math import floor
from graphqt.GUViewAxes import *
from pyimgalgos.HBins import HBins
import numpy as np
import math

#------------------------------

def image_to_hist_arr(arr, vmin=None, vmax=None, nbins=None) :
    
    amin = math.floor(arr.min() if vmin is None else vmin)
    amax = math.ceil (arr.max() if vmax is None else vmax)

    awid = math.fabs(amax - amin)
    if math.fabs(amin) < 0.01*awid :
        amin = -0.01*awid

    #mean, std = arr.mean(), arr.std()
    #amin, amax = mean-2*std, mean+10*std
    if amin == amax : amax += 1
    nhbins = int(amax-amin) if nbins is None else nbins

    NBINS_MIN = 2 if arr.dtype == np.int else 100
    NBINS_MAX = (1<<15) - 1

    #print 'XXX:NBINS_MAX', NBINS_MAX
    
    if nhbins>NBINS_MAX : nhbins=NBINS_MAX
    if nhbins<NBINS_MIN : nhbins=NBINS_MIN

    #print 'XXX arr.shape:\n', arr.shape
    #print 'XXX amin, amax, nhbins:\n', amin, amax, nhbins
    #print 'XXX arr.mean(), arr.std():\n', arr.mean(), arr.std()

    hb = HBins((amin,amax), nhbins)
    values = hb.bin_count(arr)

    return amin, amax, nhbins, values

#------------------------------

class GUViewHist(GUViewAxes) :
    
    def __init__(self, parent=None, rectax=QtCore.QRectF(0, 0, 1, 1), origin='DL', scale_ctl='HV', rulers='TBLR',\
                 margl=None, margr=None, margt=None, margb=None) :

        #xmin, xmax = np.amin(x), np.amax(x) 
        #ymin, ymax = np.amin(y), np.amax(y) 
        #w, h = xmax-xmin, ymax-ymin

        GUViewAxes.__init__(self, parent, rectax, origin, scale_ctl, rulers, margl, margr, margt, margb)

        self._name = self.__class__.__name__

        self.countemit = 0
        self.ibin_old = None
        
        self.do_cursor_scope = True
        if self.do_cursor_scope : self._init_cursor_scope()

        self.lst_items = []
        self.lst_hbins = []

        self.set_limits_horizontal() # sets self.amin, self.amax = None, None

        #self.connect_mean_std_updated_to(self.test_hist_mean_std_updated_reception)
        #self.disconnect_mean_std_updated_from(self.test_hist_mean_std_updated_reception)

        #self.connect_statistics_updated_to(self.test_statistics_std_reception)
        #self.disconnect_statistics_updated_from(self.test_statistics_std_reception)

        self.connect_axes_limits_changed_to(self.on_axes_limits_changed)
        #self.disconnect_axes_limits_changed_from(self.on_axes_limits_changed)


    def _init_cursor_scope(self) :
        self.pen1=QtGui.QPen(Qt.white, 0, Qt.DashLine)
        self.pen2=QtGui.QPen(Qt.black, 0, Qt.DashLine)
        #pen1.setCosmetic(True)
        #pen2.setCosmetic(True)

        ptrn = [10,10]
        self.pen1.setDashPattern(ptrn)
        self.pen2.setDashPattern(ptrn)
        #print 'pen1.dashPattern()', self.pen1.dashPattern()
        self.pen2.setDashOffset(ptrn[0])
        self.cline1i = self.scene().addLine(QtCore.QLineF(), self.pen1)
        self.cline2i = self.scene().addLine(QtCore.QLineF(), self.pen2)
        self.cline1i.setZValue(10)
        self.cline2i.setZValue(10)


    def set_style(self) :
        GUViewAxes.set_style(self)
        #w.setContentsMargins(-9,-9,-9,-9)
        #self.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint)
        #self.setAttribute(Qt.WA_TranslucentBackground)

#------------------------------

    def set_limits_horizontal(self, amin=None, amax=None) :
        self.amin, self.amax = amin, amax
        rax = self.rectax
        if self.amin is not None : rax.setLeft(self.amin)
        if self.amax is not None : rax.setRight(self.amax)


    def set_limits(self):
        '''Set vertical limits on histogram image in the current axes rect
        '''
        rax = self.rectax
        x1, x2 = rax.left(),   rax.right()
        y1, y2 = rax.bottom(), rax.top()

        #print 'ax x1, x2, y1, y2:', x1, x2, y1, y2
        #self.set_limits_horizontal(x1, x2) # sets self.amin, self.amax = x1, x2

        ymin, ymax = None, None

        for hb in self.lst_hbins :
            i1,i2 = hb.bin_indexes((x1,x2))
            hmin = hb.values[i1] if i1==i2 else hb.values[i1:i2].min()            
            hmax = hb.values[i1] if i1==i2 else hb.values[i1:i2].max()
            hmean= hb.values[i1] if i1==i2 else hb.values[i1:i2].mean()
            if hmax>20*hmean : hmax = 20*hmean

            ymin = hmin #if ymin is None else min(hmin,ymin) 
            ymax = hmax + 0.12*(hmax-hmin) #if ymax is None else max(hmax,ymax) 
            #print hb.values

        #print 'i1, i2, hmin, hmax:', i1, i2, hmin, hmax

        self.set_limits_vertical(ymin, ymax)

        self.evaluate_hist_mean_std()

#------------------------------

    def visible_hist_vce(self, ihis=0):
        '''Returns arrays of values, bin centers, and edges for visible part of the hystogram
        '''
        rax = self.rectax
        x1, x2 = rax.left(),   rax.right()
        y1, y2 = rax.bottom(), rax.top()

        #print 'ax x1, x2, y1, y2:', x1, x2, y1, y2
        if len(self.lst_hbins) < ihis+1 : return None

        hb = self.lst_hbins[ihis]
        i1,i2 = hb.bin_indexes((x1,x2))

        if i1 == i2 : i2+=1
        #    if i2<hb.nbins()-1 : i2+=1
        #    else               : i1-=1

        #from pyimgalgos.GlobalUtils import print_ndarr
        #print_ndarr(hb.values, name='XXX: hb.values()', first=0, last=10)
        #print_ndarr(hb.bincenters(), name='XXX: hb.bincenters()', first=0, last=10)
        #print_ndarr(hb.binedges(), name='XXX: hb.binedges()', first=0, last=10)
       
        values  = hb.values[i1:i2]
        centers = hb.bincenters()[i1:i2]
        edges   = hb.binedges()[i1:i2+1] 

        #print 'XXX: GUViewHist.visible_hist_vce'
        #print 'XXX: hb.nbins()', hb.nbins()
        #print 'XXX: i1,i2', i1,i2
        #print 'XXX: values', values
        #print 'XXX: edges', edges

        return values, centers, edges


    def visible_hist_stat(self, ihis=0):
        '''Returns statistical parameters of visible part of the hystogram
        '''
        from graphqt.GUUtils import proc_stat

        values, centers, edges = self.visible_hist_vce(ihis)

        mean, rms, err_mean, err_rms, neff, skew, kurt, err_err, sum_w = proc_stat(values,edges)
        return mean, rms, err_mean, err_rms, neff, skew, kurt, err_err, sum_w


    def visible_hist_mean_std(self, ihis=0):
        '''Returns mean and std for visible part of the histogram
        '''
        from graphqt.GUUtils import proc_stat_v2

        values, centers, edges = self.visible_hist_vce(ihis)
        mean, std = proc_stat_v2(values, centers)

        #from pyimgalgos.GlobalUtils import print_ndarr
        #print_ndarr(values, name='XXX: values', first=0, last=10)
        #print_ndarr(centers, name='XXX: centers', first=0, last=10)
        #print 'XXX visible_hist_mean_std:', mean, std
        return mean, std

#------------------------------

    def on_axes_limits_changed(self, x1, x2, y1, y2):
        #print 'XXX:GUViewHist.on_axes_limits_changed  x1: %.2f  x2: %.2f  y1: %.2f  y2: %.2f' % (x1, x2, y1, y2)      
        self.set_limits_horizontal(amin=x1, amax=x2)
        self.evaluate_hist_statistics()

#------------------------------

    def evaluate_hist_statistics(self):
        """Evaluates histogram statistical parameters and emits them with signal"""
        mean, rms, err_mean, err_rms, neff, skew, kurt, err_err, sum_w = self.visible_hist_stat()
        #self.test_statistics_std_reception(mean, rms, err_mean, err_rms, neff, skew, kurt, err_err, sum_w)
        self.emit(QtCore.SIGNAL('statistics_updated(float,float,float,float,float,float,float,float,float)'),\
                  mean, rms, err_mean, err_rms, neff, skew, kurt, err_err, sum_w)

    def connect_statistics_updated_to(self, recip) :
        self.connect(self, QtCore.SIGNAL('statistics_updated(float,float,float,float,float,float,float,float,float)'), recip)

    def disconnect_statistics_updated_from(self, recip) :
        self.disconnect(self, QtCore.SIGNAL('statistics_updated(float,float,float,float,float,float,float,float,float)'), recip)

    def test_statistics_std_reception(self, mean, rms, err_mean, err_rms, neff, skew, kurt, err_err, sum_w) :
        print 'GUViewHist.test_statistics_std_reception: ',\
              'mean, rms, err_mean, err_rms, neff, skew, kurt, err_err, sum_w\n',\
               mean, rms, err_mean, err_rms, neff, skew, kurt, err_err, sum_w

#------------------------------

    def evaluate_hist_mean_std(self):
        """Evaluates histogram mean and std (standard deviation) and emits them with signal"""
        mean, rms = self.visible_hist_mean_std()
        self.countemit+=1
        #print '%5d  mean: %.2f  rms: %.2f' % (self.countemit, mean, rms)
        self.emit(QtCore.SIGNAL('mean_std_updated(float,float)'), mean, rms)

    def connect_mean_std_updated_to(self, recip) :
        self.connect(self, QtCore.SIGNAL('mean_std_updated(float,float)'), recip)

    def disconnect_mean_std_updated_from(self, recip) :
        self.disconnect(self, QtCore.SIGNAL('mean_std_updated(float,float)'), recip)

    def test_hist_mean_std_updated_reception(self, mean, rms) :
        print 'GUViewHist.test_hist_mean_std_updated_reception mean: %.2f  rms: %.2f' % (mean, rms)

#------------------------------

    def hist_bin_value(self, x, ihis=0):
        '''Returns arrays of values, bin centers, and edges for visible part of the hystogram
        '''
        if len(self.lst_hbins) < ihis+1 : return None
        hb = self.lst_hbins[ihis]
        ind = hb.bin_index(x)
        return ind, hb.values[ind]

#------------------------------
 
    def mouseReleaseEvent(self, e):
        GUViewAxes.mouseReleaseEvent(self, e)
        #print 'GUViewHist.mouseReleaseEvent'

#------------------------------
 
    def closeEvent(self, e):
        #print 'GUViewHist.closeEvent'
        self.lst_items = []
        self.lst_hbins = []
        GUViewAxes.closeEvent(self, e)
        #print '%s.closeEvent' % self._name

#------------------------------

    def mouseMoveEvent(self, e):
        GUViewAxes.mouseMoveEvent(self, e) # calls display_pixel_pos(e)

        if self.do_cursor_scope : 
            p = self.mapToScene(e.pos())
            x, y = p.x(), p.y()
            if x<self.rectax.left() : return
            y1, y2 = self.rectax.bottom(), self.rectax.top()
            self.cline1i.setLine(x, y1, x, y2)
            self.cline2i.setLine(x, y1, x, y2)
            
            ibin, val = self.hist_bin_value(x)
            if ibin != self.ibin_old :
                #print 'x, ibin, val', x, ibin, val
                self.ibin_old = ibin
                self.emit(QtCore.SIGNAL('cursor_bin_changed(float,float,float,float)'), x, y, ibin, val)
               
        if self.pos_click is None : return
        self.set_limits()

    def connect_cursor_bin_changed_to(self, recip) :
        self.connect(self, QtCore.SIGNAL('cursor_bin_changed(float,float,float,float)'), recip)

    def disconnect_cursor_bin_changed_from(self, recip) :
        self.disconnect(self, QtCore.SIGNAL('cursor_bin_changed(float,float,float,float)'), recip)

#------------------------------

    def wheelEvent(self, e) :
        GUViewAxes.wheelEvent(self, e)
        self.set_limits()


    def enterEvent(self, e) :
    #    print 'enterEvent'
        GUViewAxes.enterEvent(self, e)
        if self.do_cursor_scope : 
            self.cline1i.setPen(self.pen1)
            self.cline2i.setPen(self.pen2)
        

    def leaveEvent(self, e) :
    #    print 'leaveEvent'
        GUViewAxes.leaveEvent(self, e)
        if self.do_cursor_scope : 
            self.cline1i.setPen(QtGui.QPen())
            self.cline2i.setPen(QtGui.QPen())

    def key_usage(self) :
        return 'Keys:'\
               '\n  ESC - exit'\
               '\n  R - reset original histogram'\
               '\n  N - set new histogram'\
               '\n'

    def keyPressEvent(self, e) :
        #print 'keyPressEvent, key=', e.key()         
        if   e.key() == Qt.Key_Escape :
            self.close()

        elif e.key() == Qt.Key_R : 
            self.reset_original_hist()

        elif e.key() == Qt.Key_N :
            from graphqt.FWViewImage import image_with_random_peaks
            print '%s: Test set new histogram' % self._name
            arr = image_with_random_peaks((50, 50))
            self.remove_all_graphs()
            hcolor = Qt.green # Qt.yellow Qt.blue Qt.yellow 
            self.add_array_as_hist(arr, pen=QtGui.QPen(hcolor, 0), brush=QtGui.QBrush(hcolor))
        else :
            print self.key_usage()

    
    def reset_original_hist(self) :

        #print 'GUViewHist.reset_original_hist'        

        #print 'Reset original size'
        #ihis=0
        #hb = self.lst_hbins[ihis]
        #amin, amax = hb.limits()
        #self.set_limits_horizontal(amin, amax)

        self.reset_original_size()
        self.set_limits()
        self.reset_original_size()

        #self.set_view()
        #self.update_my_scene()
        #self.check_axes_limits_changed()


    def _add_path_to_scene(self, path, pen=QtGui.QPen(Qt.yellow), brush=QtGui.QBrush()) :
        self.lst_items.append(self.scene().addPath(path, pen, brush))
        #self.update_my_scene() # ?????

#------------------------------

    def add_hist(self, values, edges, pen=QtGui.QPen(Qt.black), brush=QtGui.QBrush(), vtype=np.float) :
        nbins = len(values) #if isinstance(values, (list,tuple)) else values.size
        hb   = HBins(edges, nbins) #, vtype)
        binl = hb.binedgesleft()
        binr = hb.binedgesright()
        v0 = 0
        hb.values = values

        self.lst_hbins.append(hb)

        points = [QtCore.QPointF(binl[0], v0),]     # first point
        for bl, br, v in zip(binl, binr, values) :
            points.append(QtCore.QPointF(bl, v))
            points.append(QtCore.QPointF(br, v))
        points.append(QtCore.QPointF(binr[-1], v0)) # last point

        path = QtGui.QPainterPath()
        polygon = QtGui.QPolygonF(points)
        path.addPolygon(polygon)
        self._add_path_to_scene(path, pen, brush)

        self.set_limits_horizontal(amin=binl[0], amax=binr[-1])

        self.set_limits()
        #self.update_my_scene() # ?????
        self.check_axes_limits_changed()

        self.emit(QtCore.SIGNAL('histogram_updated()'))

#------------------------------

    def connect_histogram_updated_to(self, recip) :
        self.connect(self, QtCore.SIGNAL('histogram_updated()'), recip)

    def disconnect_histogram_updated_from(self, recip) :
        self.connect(self, QtCore.SIGNAL('histogram_updated()'), recip)

    def test_histogram_updated_reception(self) :
        print 'GUViewHist.test_histogram_updated_reception'

#------------------------------

    def add_array_as_hist(self, arr, pen=QtGui.QPen(Qt.black), brush=QtGui.QBrush(),\
                          vtype=np.float, set_hlims=True) :
        """Adds array (i.e. like image) as a histogram of intensities
        """
        amin, amax, nbins, values = image_to_hist_arr(arr) #, self.amin, self.amax)
        if set_hlims : 
            self.set_limits_horizontal(amin, amax)

        vmin, vmax = values.min(), values.max()
        vmean = values.mean()
        if vmax > 20*vmean : vmax = 20*vmean

        #print 'XXX: GUViewHist.add_array_as_hist: amin=%.1f  amax=%.1f  vmin=%.1f  vmax=%.1f' % (amin, amax, vmin, vmax)

        self.set_limits_vertical(None, vmax)
        self.raxes = QtCore.QRectF(self.amin, vmin, self.amax-self.amin, vmax-vmin) 

        #print 'XXX A'
        self.add_hist(values, (amin,amax), pen, brush, vtype)
        #print 'XXX B'
        self.reset_original_hist()


    def remove_all_graphs(self) :
        #print 'GUViewHist.remove_all_graphs len(self.lst_items)', len(self.lst_items)
        for item in self.lst_items :
            self.scene().removeItem(item)
        self.lst_items = []
        self.lst_hbins = []


    def __del__(self) :
        self.remove_all_graphs()

#------------------------------

if __name__ == "__main__" :

    import sys
    import numpy as np

    nbins = 1000
    x = np.array(range(nbins))-10.1
    x1 = 3.1415927/100 * x
    x2 = 3.1415927/200 * x
    y1 = np.sin(x1)/x1
    y2 = np.sin(x2)/x2
    #y2 = np.random.random((nbins,))

    rectax=QtCore.QRectF(0, -1.2, 100, 2.4)    
    app = QtGui.QApplication(sys.argv)
    w = GUViewHist(None, rectax, origin='DL', scale_ctl='H', rulers='DL',\
                    margl=0.12, margr=0.02, margt=0.02, margb=0.06)

    w.set_limits_vertical(-1.2, 1.2) 
    w.setWindowTitle("GUViewHist")
    
    w.connect_axes_limits_changed_to(w.test_axes_limits_changed_reception)
    #w.disconnect_axes_limits_changed_from(w.test_axes_limits_changed_reception)

    w.add_hist(y1, (0,100), pen=QtGui.QPen(Qt.blue, 0), brush=QtGui.QBrush(Qt.green))
    w.add_hist(y2, (0,100), pen=QtGui.QPen(Qt.red,  0), brush=QtGui.QBrush(Qt.yellow))

    #w.connect_mean_std_updated_to(w.test_hist_mean_std_updated_reception)
    #w.disconnect_mean_std_updated_from(w.test_hist_mean_std_updated_reception)

    #w.connect_statistics_updated_to(w.test_statistics_std_reception)
    #w.disconnect_statistics_updated_from(w.test_statistics_std_reception)

    w.connect_histogram_updated_to(w.test_histogram_updated_reception)
    #w.disconnect_histogram_updated_from(w.test_histogram_updated_reception)

    w.show()
    app.exec_()

#-----------------------------
