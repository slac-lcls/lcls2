
"""
Class :py:class:`GWViewColorBar` is a GWView for interactive color bar
======================================================================

GWView <- QGraphicsView <- ... <- QWidget

Usage ::

    # Create object
    #--------------
    from psana.graphqt.GWViewColorBar import GWViewColorBar
    import psana.graphqt.ColorTable as ct

    ctab = ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20)
    w = GWViewColorBar(None, coltab=ctab, orient='H', wlength=200, wwidth=50, change_mode=0o3, scale_ctl='')

    # Main methods
    #-------------
    w.set_colorbar_table_ind(ctab_ind=None) # None for next in the loop
    w.set_colorbar_table(ctab)

    ctab = w.color_table()          # current color table used in colorbar
    ctab_ind = w.color_table_index()  # current color table index

    w.connect_new_color_table_index_is_selected(recip)
    w.disconnect_new_color_table_index_is_selected(recip)
    w.connect_new_color_table_index_is_selected(w.test_new_color_table_index_is_selected_reception)

    w.connect_new_color_table(w.test_new_color_table_reception)
    w.disconnect_new_color_table(recip)
    w.connect_new_color_table(w.test_new_color_table_reception)

    w.connect_mouse_move_event(w.test_mouse_move_event_reception) # from GWViewImage

See:
    - :class:`GWView`
    - :class:`GWViewImage`
    - :class:`GWViewAxis`
    - :class:`GWViewColorBar`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2017-11-08 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-20
"""

import logging
logger = logging.getLogger(__name__)

import psana.graphqt.ColorTable as ct
from psana.graphqt.GWViewImage import *
from psana.graphqt.QWPopupSelectColorBar import popup_select_color_table
import numpy as np


class GWViewColorBar(GWViewImage):

    new_color_table = pyqtSignal()
    new_color_table_index_is_selected = pyqtSignal('int')

    def __init__(self, parent=None,\
                 coltab=ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20),\
                 orient='H', wlength=200, wwidth=50, change_mode=0o3, scale_ctl='', **kwa):
        self.orient = orient
        #print('XXX GWViewColorBar.orient: %s' % self.orient)
        arrct = ct.array_for_color_bar(coltab, orient)
        self._ctab_ind = None
        self._ctab = coltab
        self.wlength = wlength
        self.wwidth  = wwidth
        self.change_mode = change_mode
        signal_fast  = kwa.get('signal_fast', True)
        GWViewImage.__init__(self, parent, arrct, coltab=None, origin='UL', scale_ctl=scale_ctl, signal_fast=signal_fast)
        self._name  = self.__class__.__name__


    def set_style(self):
        """Overrides method from GWViewImage or GWViewImage"""
        GWViewImage.set_style(self)
        logger.debug('GWViewColorBar.set_style')

        if self.orient=='H':
            self.setMinimumSize(self.wlength, 2)
            self.setFixedHeight(self.wwidth)
        else:
            self.setMinimumSize(2, self.wlength)
            self.setFixedWidth(self.wwidth)

        #self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)


    def mousePressEvent(self, e):
        if self.change_mode & 2: self.on_colorbar(e)
        else: GWViewImage.mousePressEvent(self, e)


    def on_colorbar(self, e):
        ctab_ind = popup_select_color_table(None)
        #logger.debug('on_colorbar - selected index %s'% str(ctab_ind))
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
        self.set_pixmap_from_arr(arr, amin=arr[0], amax=arr[-1]) # method of the GWViewImage
        #self.set_style()
        self.new_color_table.emit()


    def color_table(self):
        return self._ctab


    def color_table_index(self):
        return self._ctab_ind


    def connect_new_color_table_index_is_selected(self, recip):
        self.new_color_table_index_is_selected['int'].connect(recip)

    def disconnect_new_color_table_index_is_selected(self, recip):
        self.new_color_table_index_is_selected['int'].disconnect(recip)

    def test_new_color_table_index_is_selected_reception(self, ind):
        logger.info(sys._getframe().f_code.co_name + ' color table index %s' % str(self._ctab_ind))

    def connect_new_color_table(self, recip):
        self.new_color_table_index_is_selected.connect(recip)

    def disconnect_new_color_table(self, recip):
        self.new_color_table_index_is_selected.disconnect(recip)

    def test_new_color_table_reception(self):
        logger.info(sys._getframe().f_code.co_name + ': %s' % str(self._ctab[:5]))


    def closeEvent(self, e):
        pass
        #print('%s.closeEvent' % self._name)
        #QWidget.closeEvent(self, e)


if __name__ == "__main__":
    import sys
    sys.exit(qu.msg_on_exit())

# EOF
