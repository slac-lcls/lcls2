#!@PYTHON@
"""
Class :py:class:`IVMain` is a QWidget for interactive image
===========================================================

Usage ::

    import sys
    from PyQt4 import QtGui
    from graphqt.IVMain import IVMain
    app = QtGui.QApplication(sys.argv)
    w = IVMain(None, app)
    w.show()
    app.exec_()

See:
    - :class:`IVMain`
    - :class:`IVMainTabs`
    - :class:`IVMainButtons`
    - :class:`IVImageCursorInfo`
    - :class:`IVConfigParameters`
    - :class:`IVTabDataControl`
    - :class:`IVTabFileName`
    - `graphqt documentation <https://lcls-psana.github.io/graphqt/py-modindex.html>`_.

Created on February 1, 2017 by Mikhail Dubrovin
"""
#import os
#import math

from math import floor
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt

from graphqt.IVConfigParameters import cp

from graphqt.Logger import log
import graphqt.ColorTable as ct
from graphqt.GUViewImage import GUViewImage, image_with_random_peaks
from graphqt.QWSpectrum import QWSpectrum
from graphqt.IVMainTabs import IVMainTabs
from graphqt.IVMainButtons import IVMainButtons
from graphqt.IVImageCursorInfo import IVImageCursorInfo
from graphqt.QWLogger import QWLogger
#from graphqt.IVLogger import IVLogger

from graphqt.QWUtils import selectFromListInPopupMenu

from graphqt.Frame  import Frame
from graphqt.QIcons import icon
from graphqt.Styles import style

#------------------------------

#class IVMain(Frame) :
class IVMain(QtGui.QWidget) :

    _name = 'IVMain'

    def __init__(self, parser=None) : # **dict_opts) :
        #Frame.__init__(self, parent=None, mlw=1)
        QtGui.QWidget.__init__(self, parent=None)
        #self._name = self.__class__.__name__


        self.nssd = 0


        cp.ivmain = self
        self.improd = None

        self.proc_parser(parser)
            
        self.main_win_width  = cp.main_win_width 
        self.main_win_height = cp.main_win_height
        self.main_win_pos_x  = cp.main_win_pos_x 
        self.main_win_pos_y  = cp.main_win_pos_y  
        self.color_table_ind = cp.color_table_ind 

        self.arr = self.get_image_array()

        #ctab = ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20)
        ctab = ct.next_color_table(self.color_table_ind.value())
        self.wimg = GUViewImage(None, self.arr, coltab=ctab, origin='UL', scale_ctl='HV', rulers='DL',\
                                margl=0.09, margr=0.0, margt=0.0, margb=0.04)

        self.wspe = QWSpectrum(None, self.arr, coltab=ctab, show_buts=False)

        #icon.set_icons()

        self.wtab = IVMainTabs()
        self.wbut = IVMainButtons()
        self.wcur = IVImageCursorInfo()
        self.wlog = QWLogger(log, cp, show_buttons=False)

        self.vbox = QtGui.QVBoxLayout() 
        self.vbox.addWidget(self.wtab) 
        self.vbox.addStretch(1)
        self.vbox.addWidget(self.wbut) 
        self.vbox.addWidget(self.wcur) 

        self.wrig = QtGui.QWidget()
        self.wrig.setLayout(self.vbox)

        self.vspl = QtGui.QSplitter(QtCore.Qt.Vertical)
        self.vspl.addWidget(self.wrig) 
        self.vspl.addWidget(self.wspe) 
        self.vspl.addWidget(self.wlog) 

        self.hspl = QtGui.QSplitter(QtCore.Qt.Horizontal)
        self.hspl.addWidget(self.wimg)
        self.hspl.addWidget(self.vspl)
        #self.hspl.addWidget(self.wrig)

        self.mbox = QtGui.QHBoxLayout() 
        self.mbox.addWidget(self.hspl)
        self.setLayout(self.mbox)

        self.set_style()
        #self.set_tool_tips()

        self.connect_signals_to_slots()

        #self.spectrum_show(self.arr)
        self.move(self.pos()) # + QtCore.QPoint(self.width()+5, 0))
        #self.wimg.show()

        self.on_but_reset()


    def connect_signals_to_slots(self):

        self.connect(self.wbut.but_reset, QtCore.SIGNAL('clicked()'), self.on_but_reset)
        self.connect(self.wbut.but_save,  QtCore.SIGNAL('clicked()'), self.on_but_save)

        #self.wspe.connect_color_table_is_changed_to(self.on_spectrum_color_table_is_changed)
        self.wspe.cbar.connect_new_color_table_to(self.on_spectrum_color_table_is_changed)

        self.connect_signals_from_img()
        self.connect_signals_from_hist()

        #self.wspe.hist.connect_axes_limits_changed_to(self.wspe.hist.test_axes_limits_changed_reception)
        #self.wspe.connect_color_table_is_changed_to(self.wspe.test_color_table_is_changed_reception)
        #self.wspe.hist.connect_histogram_updated_to(self.wspe.hist.test_histogram_updated_reception)

        #self.wimg.disconnect_axes_limits_changed_from(self.wimg.test_axes_limits_changed_reception)
        #self.wimg.connect_axes_limits_changed_to(self.wimg.test_axes_limits_changed_reception)

        #self.wimg.connect_pixmap_is_updated_to(self.wimg.test_pixmap_is_updated_reception)
        #self.wimg.disconnect_pixmap_is_updated_from(self.wimg.test_pixmap_is_updated_reception)

    def connect_signals_from_img(self):
        self.wimg.connect_axes_limits_changed_to(self.on_image_axes_limits_changed)
        self.wimg.connect_pixmap_is_updated_to(self.on_image_pixmap_is_updated)
        self.wimg.connect_cursor_pos_value_to(self.wcur.set_cursor_pos_value)


    def disconnect_signals_from_img(self):
        self.wimg.disconnect_axes_limits_changed_from(self.on_image_axes_limits_changed)
        self.wimg.disconnect_pixmap_is_updated_from(self.on_image_pixmap_is_updated)
        self.wimg.disconnect_cursor_pos_value_from(self.wcur.set_cursor_pos_value)


    def connect_signals_from_hist(self):
        self.wspe.hist.connect_axes_limits_changed_to(self.on_hist_axes_limits_changed)


    def disconnect_signals_from_hist(self):
        self.wspe.hist.disconnect_axes_limits_changed_from(self.on_hist_axes_limits_changed)

#------------------------------

    def proc_parser(self, parser=None) :
        self.parser=parser

        if parser is None :
            cp.fname_img.setValue('')
            return

        (popts, pargs) = parser.parse_args()
        self.args = pargs
        self.opts = vars(popts)
        self.defs = vars(parser.get_default_values())

        nargs =len(self.args)

        exp = popts.exp # self.opts['exp']
        run = popts.run # self.opts['run']
        nev = popts.nev
        clb = popts.clb
        ifn = popts.ifn
        vrb = popts.vrb

        #cp.instr_dir .setValue() # val_def='/reg/d/psdm'
        if exp != self.defs['exp'] : cp.instr_name.setValue(exp[:3].upper())
        if exp != self.defs['exp'] : cp.exp_name  .setValue(exp)
        if run != self.defs['run'] : cp.str_runnum.setValue('%d'%run)
        if clb != self.defs['clb'] : cp.calib_dir .setValue(clb)

        self.verbos = vrb
 
        ifname = ifn          if ifn != self.defs['ifn'] else\
                 self.args[0] if nargs > 0 else\
                 None
        
        if ifname is not None :
            log.info('Input image file name: %s' % ifname)
            cp.fname_img.setValue(ifname)
            cp.current_tab.setValue('File')
        #else :
        #    cp.current_tab.setValue('Data')

#------------------------------

    def get_image_array(self) :
        import expmon.PSUtils as psu
        ifname = cp.fname_img.value()
        arr = psu.get_image_array_from_file(ifname)
        if arr is None :
            log.warning('%s Can not get image from file: %s Substitute simulated image' % (self._name, ifname))
            arr = image_with_random_peaks((1000, 1000))

        log.info('Image array shape: %s' % str(arr.shape))
        return arr
    
#------------------------------

    def print_pars(self) :
        """Prints input parameters"""
        print 'In print_pars:' 
        for k,v in self.opts.items() :
            print '%s %s %s' % (k.ljust(10), str(v).ljust(16), str(self.defs[k]).ljust(16))

#------------------------------

    def spectrum_show(self, arr=None):
        a = self.arr if arr is None else arr
        #self.wspe.move(self.pos() + QtCore.QPoint(self.width(), 0))
        #self.wspe.show()


    def spectrum_close(self):
        pass
        #self.wspe.hist.disconnect_axes_limits_changed_from(self.wspe.hist.test_axes_limits_changed_reception)
        ##self.wspe.hist.disconnect_histogram_updated_from(self.wspe.hist.test_histogram_updated_reception)
        #self.wspe.disconnect_color_table_is_changed_from(self.wspe.test_color_table_is_changed_reception)

        #if self.wspe is None : return
        #try :
        #    self.wspe.close()
        #except : pass
        #self.wspe = None


    def set_image_data(self, arr, set_hlims=False):
        '''Sets new image data array:
        '''
        log.info('%s.set_image_data' % self._name)
        if arr is None :
            log.warning('%s.set_image_data: data array is None' % (self._name))
            return

        self.wimg.set_pixmap_from_arr(arr)
        self.set_spectral_data(arr, set_hlims=set_hlims)


    def set_spectral_data(self, arr, hcolor = Qt.green, set_hlims=True):
        log.debug('%s.set_spectral_data, set_hlims=%s' % (self._name,set_hlims))
        #self.nssd+=1; print 'XXX %2d %s.set_spectral_data, set_hlims=%s' % (self.nssd, self._name, set_hlims)
        self.wspe.hist.remove_all_graphs()
        self.wspe.hist.add_array_as_hist(arr, pen=QtGui.QPen(hcolor, 0),\
                                         brush=QtGui.QBrush(hcolor), set_hlims=set_hlims)


    def on_but_reset(self):
        log.info('%s.on_but_reset' % self._name)
        self.is_reset=True

        self.wimg.on_but_reset()

        rax = self.wimg.rect_axes()
        #rax.moveCenter(rax.center() + QtCore.QPointF(1, 0))
        self.wimg.set_rect_axes(rax) # need to call it to update image window->reset spectrum

        self.is_reset=False


    def on_but_save(self):
        #log.info('%s.on_but_save' % self._name)
        slst = ['Spectrum', 'Image', 'Both']
        sel = selectFromListInPopupMenu(slst)
        if sel is None : return
        if sel in (slst[0],slst[2]) : self.wspe.on_but_save()
        if sel in (slst[1],slst[2]) : self.wimg.on_but_save(at_obj=self.wbut.but_save)


    def on_spectrum_color_table_is_changed(self):
        '''Responce on signal color_table_is_changed():
        '''
        log.info('%s.on_color_table_is_changed' % self._name)
        #self.wimg.set_color_table(self.wspe.color_table())
        self.wimg.set_color_table(self.wspe.cbar.color_table())

        self.wimg.set_pixmap_from_arr()


    def on_hist_axes_limits_changed(self, x1, x2, y1, y2):
        '''Responce on signal axes_limits_changed():
        '''
        log.info('%s.on_hist_axes_limits_changed x1: %.2f  x2: %.2f  y1: %.2f  y2: %.2f'%\
                 (self._name, x1, x2, y1, y2))

        self.disconnect_signals_from_hist()

        self.wimg.set_intensity_limits(amin=x1, amax=x2)
        self.wimg.set_pixmap_from_arr()

        self.connect_signals_from_hist()


    def on_image_axes_limits_changed(self, x1, x2, y1, y2) :
        '''Responce on signal.
        '''
        #log.info('%s.on_image_axes_limits_changed x1: %.2f  x2: %.2f  y1: %.2f  y2: %.2f'%\
        #         (self._name, x1, x2, y1, y2))
        arr = self.wimg.image_data()
        h, w = arr.shape
        h1, w1 = h-1, w-1

        xmin, xmax = int(floor(min(x1, x2))), int(floor(max(x1, x2)))
        ymin, ymax = int(floor(min(y1, y2))), int(floor(max(y1, y2)))

        if xmin<0 : xmin = 0
        if ymin<0 : ymin = 0
        if xmax<0 : xmax = 0
        if ymax<0 : ymax = 0

        if xmin>w1: xmin = w1
        if ymin>h1: ymin = h1
        if xmax>w1: xmax = w1
        if ymax>h1: ymax = h1

        arr_win = arr[ymin:ymax, xmin:xmax]

        #log.info('image data shape h=%d w=%d' % (h,w))
        log.info('%s.on_image_axes_limits_changed xmin: %4d  xmax: %4d  ymin: %4d  ymax: %4d'%\
                (self._name, xmin, xmax, ymin, ymax))

        self.disconnect_signals_from_img()

        self.set_spectral_data(arr_win, hcolor = Qt.yellow, set_hlims=self.is_reset)

        self.connect_signals_from_img()


    def on_image_pixmap_is_updated(self):
        '''Responce on signal:
        '''
        log.debug('%s.on_image_pixmap_is_updated' % self._name)


    def on_image_file_is_changed(self, fname):
        '''Responce on signal from IVFileName
        '''
        print 'YYY: IVMain.on_image_file_is_changed %s'%fname
        self.arr = self.get_image_array()
        #h,w = self.arr.shape
        #print 'ZZZ:shape', self.arr.shape
        #self.on_image_axes_limits_changed(0, w, 0, h)
        self.set_image_data(self.arr, set_hlims=True)
        #self.on_but_reset()
        #self.wimg.update_my_scene()


    def on_new_event_number(self, num):
        '''Responce on signal from QWEventControl -> QWDataControl
        '''
        log.debug('%s.on_new_event_number %d' % (self._name, num))

        set_hlims=False
        if self.improd is None : 
            from expmon.PSImageProducer import PSImageProducer
            self.improd = PSImageProducer(cp, log)
            set_hlims=True

        self.arr = self.improd.image(num)
        self.set_image_data(self.arr, set_hlims)


    def set_tool_tips(self):
        pass
        #self.butStop.setToolTip('Not implemented yet...')


    def set_style(self):
        #self.setGeometry(50, 50, 500, 600)
        self.setGeometry(self.main_win_pos_x .value(),\
                         self.main_win_pos_y .value(),\
                         self.main_win_width .value(),\
                         self.main_win_height.value())

        self.setMinimumSize(1200, 700)

        w = self.main_win_width.value()

        self.setContentsMargins(QtCore.QMargins(-9,-9,-9,-9))
        self.wspe.setFixedHeight(280)
        self.wimg.setMinimumWidth(700)
        self.wimg.setSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)

        self.wrig.setContentsMargins(-9,-9,-9,-9)
        self.wrig.setMinimumWidth(350)
        self.wrig.setMaximumWidth(450)

        #self.wrig.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Ignored)
        #self.hspl.moveSplitter(w*0.5,0)

        #self.setFixedSize(800,500)
        #self.setMinimumSize(500,800)

        #self.setStyleSheet("background-color:blue; border: 0px solid green")
        #self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        
        #self.        setStyleSheet(style.styleBkgd)
        #self.butSave.setStyleSheet(style.styleButton)
        #self.butExit.setStyleSheet(style.styleButton)
        #self.butELog.setStyleSheet(style.styleButton)
        #self.butFile.setStyleSheet(style.styleButton)

        #self.butELog    .setVisible(False)
        #self.butFBrowser.setVisible(False)

        #self.but1.raise_()


    def closeEvent(self, e):
        log.debug('%s.closeEvent' % self._name)
        try : self.wimg.close()
        except : pass

        try : self.wspe.close()
        except : pass

        self.on_save()

        QtGui.QWidget.closeEvent(self, e)
        cp.ivmain = None

 
    def resizeEvent(self, e):
        #log.debug('resizeEvent', self._name) 
        #log.info('IVMain.resizeEvent: %s' % str(self.size()))
        pass


    def moveEvent(self, e):
        #log.debug('moveEvent', self._name) 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #log.debug('moveEvent - pos:' + str(self.position), __name__)       
        #log.info('IVMain.moveEvent - move window to x,y: ', str(self.mapToGlobal(QtCore.QPoint(0,0))))
        #self.wimg.move(self.pos() + QtCore.QPoint(self.width()+5, 0))
        pass


    def keyPressEvent(self, e) :
        log.info('%s.keyPressEvent, key=%d' % (self._name, e.key()))         
        if   e.key() == Qt.Key_Escape :
            self.close()

        elif e.key() == Qt.Key_U : 
            log.info('%s: Test set new image' % self._name)
            img = image_with_random_peaks((1000, 1000))
            self.set_image_data(img)



    def on_save(self):

        point, size = self.mapToGlobal(QtCore.QPoint(-5,-22)), self.size() # Offset (-5,-22) for frame size.
        x,y,w,h = point.x(), point.y(), size.width(), size.height()
        msg = 'Save main window x,y,w,h : %d, %d, %d, %d' % (x,y,w,h)
        log.info(msg, self._name)
        #print msg

        #Save main window position and size
        self.main_win_pos_x .setValue(x)
        self.main_win_pos_y .setValue(y)
        self.main_win_width .setValue(w)
        self.main_win_height.setValue(h)

        self.color_table_ind.setValue(ct.STOR.ictab)

        cp.printParameters()
        cp.saveParametersInFile()

        if cp.save_log_at_exit.value() :
            log.saveLogInFile(cp.log_file.value())
            #print 'Saved log file: %s' % cp.log_file.value()
            #log.saveLogTotalInFile(fnm.log_file_total())


#------------------------------
if __name__ == "__main__" :
    import sys

    log.setPrintBits(0377) 

    app = QtGui.QApplication(sys.argv)
    ex  = IVMain(parser=None)
    ex.show()
    app.exec_()
#------------------------------
