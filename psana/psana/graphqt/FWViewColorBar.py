"""
Class :py:class:`FWViewColorBar` is a FWView for interactive color bar
======================================================================

FWView <- QGraphicsView <- ... <- QWidget

Usage ::

    # Create object
    #--------------
    from psana.graphqt.FWViewColorBar import FWViewColorBar
    import psana.graphqt.ColorTable as ct

    ctab = ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20)
    w = FWViewColorBar(None, coltab=ctab, orient='H', wlength=200, wwidth=50, change_mode=0o3, scale_ctl='')

    # Main methods
    #-------------
    w.set_colorbar_table_ind(ctab_ind=None) # None for next in the loop
    w.set_colorbar_table(ctab)

    ctab = w.color_table()          # current color table used in colorbar
    ctab_ind = w.color_table_index()  # current color table index

    w.connect_new_color_table_index_is_selected_to(recip)
    w.disconnect_new_color_table_index_is_selected_from(recip)
    w.connect_new_color_table_index_is_selected_to(w.test_new_color_table_index_is_selected_reception)

    w.connect_new_color_table_to(w.test_new_color_table_reception)
    w.disconnect_new_color_table_from(recip)
    w.connect_new_color_table_to(w.test_new_color_table_reception)

    w.connect_mouse_move_event_to(w.test_mouse_move_event_reception) # from FWViewImage

See:
    - :class:`FWView`
    - :class:`FWViewImage`
    - :class:`FWViewAxis`
    - :class:`FWViewColorBar`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2017-11-08 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-20
"""

import ColorTable as ct
from FWViewImage import *
from QWPopupSelectColorBar import popup_select_color_table
import numpy as np

#----

class FWViewColorBar(FWViewImage):

    new_color_table = pyqtSignal()
    new_color_table_index_is_selected = pyqtSignal('int')

    def __init__(self, parent=None,\
                 coltab=ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20),\
                 orient='H', wlength=200, wwidth=50, change_mode=0o3, scale_ctl=''):
        self.orient = orient
        #print('XXX FWViewColorBar.orient: %s' % self.orient)
        arrct = ct.array_for_color_bar(coltab, orient)
        self._ctab_ind = None
        self._ctab = coltab
        self.wlength = wlength
        self.wwidth  = wwidth
        self.change_mode = change_mode
        FWViewImage.__init__(self, parent, arrct, coltab=None, origin='UL', scale_ctl=scale_ctl)
        self._name  = self.__class__.__name__


    def set_style(self):
        """Overrides method from FWViewImage or FWViewImage
        """
        FWViewImage.set_style(self)
        if self.orient=='H':
            self.setMinimumSize(self.wlength, 2)
            self.setFixedHeight(self.wwidth)
        else:
            self.setMinimumSize(2, self.wlength)
            self.setFixedWidth(self.wwidth)

        #self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)


    def mousePressEvent(self, e):
        if self.change_mode & 2: self.on_colorbar(e)
        else: FWViewImage.mousePressEvent(self, e)


    def on_colorbar(self, e):
        #print('QWSpectrum.on_colorbar')
        ctab_ind = popup_select_color_table(None)
        if ctab_ind is None: return
        if ctab_ind != self._ctab_ind:
            self._ctab_ind = ctab_ind
            self.set_colorbar_table_ind(ctab_ind)


    def set_colorbar_table_ind(self, ctab_ind=None):
        """Sets color table from the list of pre-set tables by their index.
           ctab_ind=None - set next table in a loop.
        """
        ctab = ct.next_color_table(ctab_ind)
        self._ctab_ind = ctab_ind if ctab_ind is not None else ct.STOR.color_table_index()
        self.set_colorbar_table(ctab)
        #self.emit(QtCore.SIGNAL('new_color_table_index_is_selected(int)'), self._ctab_ind)
        self.new_color_table_index_is_selected.emit(self._ctab_ind)


    def set_colorbar_table(self, ctab):
        """Sets color table ctab (np.array) - list of colors 32-bit unsigned words"""
        self._ctab = ct.np.array(ctab)
        arr = ct.array_for_color_bar(ctab, self.orient)
        self.set_pixmap_from_arr(arr) # method of the FWViewImage
        #print('XXX Color table:', ctab)
        #self.emit(QtCore.SIGNAL('new_color_table()'))
        self.new_color_table.emit()


    def color_table(self):
        return self._ctab

 
    def color_table_index(self):
        return self._ctab_ind


    def connect_new_color_table_index_is_selected_to(self, recip):
        self.new_color_table_index_is_selected['int'].connect(recip)
        #self.connect(self, QtCore.SIGNAL('new_color_table_index_is_selected(int)'), recip)


    def disconnect_new_color_table_index_is_selected_from(self, recip):
        self.new_color_table_index_is_selected['int'].disconnect(recip)
        #self.disconnect(self, QtCore.SIGNAL('new_color_table_index_is_selected(int)'), recip)


    def connect_new_color_table_to(self, recip):
        self.new_color_table_index_is_selected.connect(recip)
        #self.connect(self, QtCore.SIGNAL('new_color_table()'), recip)


    def disconnect_new_color_table_from(self, recip):
        self.new_color_table_index_is_selected.disconnect(recip)
        #self.disconnect(self, QtCore.SIGNAL('new_color_table()'), recip)


    def test_new_color_table_reception(self):
        print('  FWViewColorBar.test_new_color_table_reception: %s' % str(self._ctab[:5]))


    def test_new_color_table_index_is_selected_reception(self, ind):
        print('  FWViewColorBar.test_new_color_table_index_is_selected_reception: %s' % str(self._ctab_ind))


    def closeEvent(self, e):
        pass
        #print('%s.closeEvent' % self._name)
        #QWidget.closeEvent(self, e)

#----

    if __name__ == "__main__":

      def key_usage(self):
        return 'Keys:'\
               '\n  ESC - exit'\
               '\n  R - reset color table 0'\
               '\n  N - set next color table'\
               '\n'

      def keyPressEvent(self, e):
        #print('keyPressEvent, key=', e.key())

        if not (self.change_mode & 1): return

        if   e.key() == Qt.Key_Escape:
            self.close()

        elif e.key() == Qt.Key_R: 
            print('Reset original size')
            self.set_colorbar_table_ind(ctab_ind=0)

        elif e.key() == Qt.Key_N: 
            print('Set next color table')
            self.set_colorbar_table_ind(ctab_ind=None)

        else:
            print(self.key_usage())
  
#----

if __name__ == "__main__":

  import sys

  def test_wfviewcolorbar(tname):
    print('%s:' % sys._getframe().f_code.co_name)
    arr = np.random.random((1000, 100))
    #arr = image_with_random_peaks((1000, 1000))
    ctab = ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20)
    #ctab = ct.color_table_monochr256()
    #ctab = ct.color_table_interpolated()

    app = QApplication(sys.argv)
    w = None
    if   tname == '0': w = FWViewColorBar(None, coltab=ctab, orient='H', change_mode=1, scale_ctl='H')
    elif tname == '1': w = FWViewColorBar(None, coltab=ctab, orient='V')
    else:
        print('test %s is not implemented' % tname)
        return

    w.setWindowTitle(w._name)

    w.connect_mouse_move_event_to(w.test_mouse_move_event_reception)
    w.connect_new_color_table_index_is_selected_to(w.test_new_color_table_index_is_selected_reception)
    w.connect_new_color_table_to(w.test_new_color_table_reception)

    w.show()
    app.exec_()

    del w
    del app

#----

if __name__ == "__main__":
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print(50*'_', '\nTest %s' % tname)
    test_wfviewcolorbar(tname)
    sys.exit('End of Test %s' % tname)

#----
