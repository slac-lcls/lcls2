#!@PYTHON@
#------------------------------
"""
Class :py:class:`QWSpectrum` - supports widget for spectrum
===========================================================

Usage ::

    Create GUViewHist object within pyqt QApplication
    --------------------------------------------------
    import sys
    from PyQt4 import QtGui, QtCore
    from graphqt.QWSpectrum import QWSpectrum

    arr = image_with_random_peaks((50, 50))
    app = QtGui.QApplication(sys.argv)
    w = QWSpectrum(None, arr, show_frame=False) #, show_buts=False)

    Connect/disconnecr recipient to signals
    ---------------------------------------

    ### w.connect_color_table_is_changed_to(recipient)
    ### w.disconnect_color_table_is_changed_from(recipient) :
    ### w.test_color_table_is_changed_reception(self)

    For self.hist:
    self.hist.connect_axes_limits_changed_to(recipient)
    self.hist.connect_histogram_updated_to(recipient)
    self.hist.disconnect_histogram_updated_from(recipient)


    Methods
    -------
    w.on_but_save()
    w.on_but_reset()
    ctab = w.color_table(self)
    w.on_colorbar() # emits signal color_table_is_changed
    w.draw_mean_std(mean, std)
    w.draw_stat(mean, rms, err_mean, err_rms, neff, skew, kurt, err_err, sum_w)
    w.draw_cursor_locator(x, y, ibin, value)
    w.set_tool_tips()
    w.set_style()

    Internal methods
    -----------------

    Re-defines methods
    ------------------
    closeEvent

    Global scope methods
    --------------------
    test_guspectrum(tname)

See:
    * :py:class:`IVMain`
    * :py:class:`IVMainTabs`
    * :py:class:`IVConfigParameters`
    * :py:class:`IVImageCursorInfo`
    * :py:class:`IVMainTabs`
    * :py:class:`IVTabDataControl`
    * :py:class:`IVTabFileName`
 
Created on 2017-02-06 by Mikhail Dubrovin
"""
#------------------------------

import os
import sys
#from graphqt.Frame import Frame
from PyQt4 import QtGui, QtCore
Qt = QtCore.Qt
from graphqt.Styles import style    

from graphqt.FWViewImage import image_with_random_peaks
from graphqt.FWViewColorBar import FWViewColorBar
from graphqt.GUViewHist import GUViewHist
import graphqt.ColorTable as ct
from graphqt.QWPopupSelectColorBar import popup_select_color_table

#------------------------------

class QWSpectrum(QtGui.QWidget) : # QtGui.QWidget, Frame
    """Widget for file name input
    """
    def __init__(self, parent=None, arr=None,\
                 coltab = ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20),
                 show_frame=True, show_buts=True) :

        QtGui.QWidget.__init__(self, parent)
        #Frame.__init__(self, parent, mlw=1, vis=show_frame)
        self._name = self.__class__.__name__
        self.show_frame = show_frame
        self.show_buts  = show_buts

        self.but_save = QtGui.QPushButton('&Save')
        self.but_reset= QtGui.QPushButton('&Reset')

        self.lab_stat = QtGui.QLabel('    Histogram\n    statistics')
        self.lab_ibin = QtGui.QLabel('Bin info')

        #amin, amax, nbins, values = image_to_hist_arr(arr)
        #vmin, vmax = values.min(), values.max()
        #rectax=QtCore.QRectF(amin, vmin, amax-amin, vmax-vmin)
        rectax=QtCore.QRectF(0, 0, 1, 1)

        self.hist = GUViewHist(None, rectax, origin='DL', scale_ctl='H', rulers='DL',
                               margl=0.12, margr=0.01, margt=0.01, margb=0.15)
        self.hist.connect_mean_std_updated_to(self.draw_mean_std)
        self.hist.connect_statistics_updated_to(self.draw_stat)
        self.hist.connect_cursor_bin_changed_to(self.draw_cursor_locator)
        self.hist._ymin = None
        #self.hist._ymax = 1.5

        hcolor = Qt.yellow # Qt.green Qt.yellow Qt.blue 
        #self.hist.add_hist(values, (amin,amax), pen=QtGui.QPen(hcolor, 0), brush=QtGui.QBrush(hcolor)) # vtype=np.float
        self.hist.add_array_as_hist(arr, pen=QtGui.QPen(hcolor, 0), brush=QtGui.QBrush(hcolor))

        #self.ctab = ct.color_table_monochr256()
        #self.ctab = ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20)
        #self.ctab = ct.color_table_interpolated()
        #self.ctab = coltab
        #arrct = ct.array_for_color_bar(self.ctab, orient='H')
        #self.cbar = FWViewImage(None, arrct, coltab=None, origin='UL', scale_ctl='') # 'H'
        self.cbar = FWViewColorBar(None, coltab=coltab, orient='H')
        #self.hist.move(10,10)
        #self.cbar.move(50,200)
        #self.label = QtGui.QLineEdit(self)
        #self.label.move(130, 22)

        #self.vbox = QtGui.QVBoxLayout() 
        #self.vbox.addWidget(self.cbar)
        #self.vbox.addWidget(self.hist)
        #self.vbox.addStretch(1)
        #self.setLayout(self.vbox)

        grid = QtGui.QGridLayout()
        grid.addWidget(self.hist,      0,  0, 100, 100)
        grid.addWidget(self.cbar,     96, 12,   4,  88)
        grid.addWidget(self.lab_stat,  0, 80,  10,  20)
        grid.addWidget(self.lab_ibin,  0, 12,   5,  20)

        grid.addWidget(self.but_reset, 92, 0,   4,  10)
        grid.addWidget(self.but_save,  96, 0,   4,  10)
        #grid.addWidget(self.cbar,  0, 13,  4,  86)
        self.setLayout(grid) 

        self.set_tool_tips()
        self.set_style()

        #self.cbar.connect_mouse_press_event_to(self.on_colorbar)

        #self.hist.disconnect_mean_std_updated_from(self.draw_stat)
        #self.hist.disconnect_statistics_updated_from(self.draw_stat)
        #self.cbar.disconnect_click_on_color_bar_from(self.on_colorbar)
        #self.connect_color_table_is_changed_to(self.test_color_table_is_changed_reception)

        if self.show_buts :
          self.connect(self.but_save,  QtCore.SIGNAL('clicked()'), self.on_but_save)
          self.connect(self.but_reset, QtCore.SIGNAL('clicked()'), self.on_but_reset)

        #self.on_but_reset()

#------------------------------
 
    def on_but_save(self) :
        fltr='*.png *.gif *.jpg *.jpeg\n *'
        fname = 'fig-spectrum.png'
        fname = str(QtGui.QFileDialog.getSaveFileName(self, 'Output file', fname, filter=fltr))
        if fname == '' : return
        print 'QWSpectrum.on_but_save: save image in file: %s' % fname
        #p = QtGui.QPixmap.grabWindow(self.winId())
        p = QtGui.QPixmap.grabWidget(self, self.rect())
        p.save(fname, 'jpg')
    
#------------------------------
 
    def on_but_reset(self) :
        #print 'QWSpectrum.on_but_reset TBD'
        self.hist.reset_original_hist()

#------------------------------
 
    def color_table(self) :
        return self.ctab

#------------------------------
 
#    def on_colorbar(self, e) :
#        #print 'QWSpectrum.on_colorbar'
#        ctab_ind = popup_select_color_table(None)
#        if ctab_ind is None : return
#        self.ctab = ct.next_color_table(ctab_ind)
#        arr = ct.array_for_color_bar(self.ctab, orient='H')
#        self.cbar.set_pixmap_from_arr(arr)
#        self.emit(QtCore.SIGNAL('color_table_is_changed()'))

#    def connect_color_table_is_changed_to(self, recip) :
#        self.connect(self, QtCore.SIGNAL('color_table_is_changed()'), recip)

#    def disconnect_color_table_is_changed_from(self, recip) :
#        self.disconnect(self, QtCore.SIGNAL('color_table_is_changed()'), recip)

#    def test_color_table_is_changed_reception(self) :
#        print 'QWSpectrum.color_table_is_changed:', self.ctab.shape

#------------------------------
 
    def draw_mean_std(self, mean, std) :
        txt = '    Mean: %.2f\n    RMS: %.2f' % (mean, std)
        #print txt
        self.lab_stat.setText(txt)

#------------------------------
 
    def draw_stat(self, mean, rms, err_mean, err_rms, neff, skew, kurt, err_err, sum_w) :
        #print 'XXX: mean, rms, err_mean, err_rms, neff, skew, kurt, err_err, sum_w',\
        #            mean, rms, err_mean, err_rms, neff, skew, kurt, err_err, sum_w
        txt = u'  Entries: %d\n  Mean: %.2f \u00B1 %.2f\n  RMS: %.2f \u00B1 %.2f\n  \u03B31=%.2f   \u03B32=%.2f'%\
              (neff, mean, err_mean, rms, err_rms, skew, kurt)
        #print txt
        self.lab_stat.setText(txt)

#------------------------------
 
    def draw_cursor_locator(self, x, y, ibin, value) :
        txt = '  Bin:%d  value=%.2f' % (ibin, value)
        #print txt
        self.lab_ibin.setText(txt)

#------------------------------

    def set_tool_tips(self) :
        #self.hist.setToolTip('Spectrum histogram')
        self.cbar.setToolTip('Color bar') 

#------------------------------

    def set_style(self) :
        self.setWindowTitle('Spectrum with color bar')

        self.setMinimumSize(400,150)
        #self.setFixedSize(750,270)
        #self.setMaximumWidth(800)
 
        #self.setGeometry(50, 50, 600, 300)
        self.setGeometry(50, 50, 500, 300)
        #self.cbar.setFixedHeight(22)
        #self.cbar.setFixedSize(600, 22)
        #self.cbar.setMinimumSize(300, 22)
        self.cbar.setMinimumSize(200, 2)
        self.cbar.setFixedHeight(22)
        #self.setMinimumWidth(300)
        #self.edi.setMinimumWidth(210)
        #self.setFixedHeight(50 if self.show_frame else 34)
        #if not self.show_frame : self.setContentsMargins(-9,-9,-9,-9)
        self.setContentsMargins(-9,-9,-9,-9)

        #style = "background-color: rgb(255, 255, 220); color: rgb(0, 0, 0);" # Yellowish
        #style = "background-color: rgb(100, 240, 200); color: rgb(0, 0, 0);" # Greenish
        #style = "background-color: rgb(255, 200, 220); color: rgb(0, 0, 0);" # Pinkish
        #style = "background-color: rgb(240, 240, 100); color: rgb(0, 0, 0);" # YellowBkg
        #self.setStyleSheet(style)

        self.lab_stat.setStyleSheet(style.styleStat)
        self.lab_ibin.setStyleSheet(style.styleStat)
        self.lab_ibin.setFixedSize(150,20)

        self.but_reset.setFixedSize(50,30)
        self.but_save .setFixedSize(50,30)

        self.but_reset.setVisible(self.show_buts)
        self.but_save .setVisible(self.show_buts)

#------------------------------

    def closeEvent(self, e):
        #log.info('closeEvent', self._name)
        #print '%s.closeEvent' % self._name

        try : self.hist.close()
        except : pass

        try : self.cbar.close()
        except : pass

        QtGui.QWidget.closeEvent(self, e)
 
#------------------------------

def test_guspectrum(tname) :
    print '%s:' % sys._getframe().f_code.co_name

    arr = image_with_random_peaks((50, 50))
    app = QtGui.QApplication(sys.argv)
    w = QWSpectrum(None, arr, show_frame=False) #, show_buts=False)

    #w.connect_color_table_is_changed_to(w.test_color_table_is_changed_reception)
    w.cbar.connect_new_color_table_to(w.cbar.test_new_color_table_reception)

    w.hist.connect_axes_limits_changed_to(w.hist.test_axes_limits_changed_reception)
    #w.hist.disconnect_axes_limits_changed_from(self.hist.test_axes_limits_changed_reception)

    w.hist.connect_histogram_updated_to(w.hist.test_histogram_updated_reception)
    #w.hist.disconnect_histogram_updated_from(w.hist.test_histogram_updated_reception)

    w.show()
    app.exec_()

#------------------------------

if __name__ == "__main__" :
    import sys; global sys
    import numpy as np; global np
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print 50*'_', '\nTest %s' % tname
    test_guspectrum(tname)
    sys.exit('End of Test %s' % tname)

#------------------------------
