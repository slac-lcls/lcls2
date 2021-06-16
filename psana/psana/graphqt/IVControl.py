
"""Class :py:class:`IVControl` is a Image Viewer QWidget with control fields
==========================================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/IVControl.py

    from psana.graphqt.IVControl import IVControl
    w = IVControl()

Created on 2021-06-14 by Mikhail Dubrovin
"""

import logging
#logger = logging.getLogger(__name__)

import sys
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout#, QComboBox, QPushButton, QLineEdit
#from PyQt5.QtCore import QSize # Qt, QEvent, QPoint, 
#from psana.graphqt.H5VConfigParameters import cp
#from psana.graphqt.QWIcons import icon
from psana.graphqt.QWFileName import QWFileName


class IVControl(QWidget):
    """QWidget for Image Viewer control fields"""

    def __init__(self, **kwargs):

        parent = kwargs.get('parent',None)
        d = '/reg/g/psdm/detector/alignment/epix10ka2m/calib-xxx-epix10ka2m.1-2021-02-02/'
        fname_nda = d + 'det-calib-mfxc00118-r242-e5000-max.txt'
        fname_geo = d + '2021-02-02-epix10ks2m.1-geometry-recentred-for-psana.txt'

        QWidget.__init__(self, parent)
        #self._name = 'IVControl'
        #self.lab_ctrl = QLabel('Control:')

        #self.but_exp_col  = QPushButton('Collapse')

        #fname = cp.h5vmain.wtree.fname if cp.h5vmain is not None else './test.h5'
        self.w_fname_nda = QWFileName(None, butname='Select', label='N-d array:',\
           path=fname_nda, fltr='*.txt *.npy\n*', show_frame=True)

        self.w_fname_geo = QWFileName(None, butname='Select', label='Geometry:',\
           path=fname_geo, fltr='*.txt *.data\n*', show_frame=True)

        self.box1 = QVBoxLayout() 
        #self.box1.addWidget(self.lab_ctrl)
        #self.box1.addWidget(self.but_exp_col)
        self.box1.addWidget(self.w_fname_geo)
        self.box1.addStretch(1) 
        self.box1.addWidget(self.w_fname_nda)
        #self.box1.addSpacing(20)

        #self.box1.addLayout(self.grid)
        self.setLayout(self.box1)
 
        #self.but_exp_col.clicked.connect(self.on_but_clicked)
        #self.but_exp_col.clicked.connect(self.on_but_exp_col)

        #if cp.h5vmain is not None:
        #    self.w_fname_nda.connect_path_is_changed_to_recipient(cp.h5vmain.wtree.set_file)

        self.set_tool_tips()
        #self.set_style()
        #self.set_buttons_visiable()


    def set_tool_tips(self):
        self.setToolTip('Control fields/buttons')


    def set_style(self):
        from psana.graphqt.Styles import style
        #self.         setStyleSheet(style.styleBkgd)
        #self.lab_db_filter.setStyleSheet(style.styleLabel)
        #self.lab_ctrl.setStyleSheet(style.styleLabel)

        icon.set_icons()
        #self.but_exp_col.setIcon(icon.icon_folder_open)


    def on_but_exp_col(self):
        if cp.h5vmain is None: return

        wtree = cp.h5vmain.wtree
        but = self.but_exp_col
        if but.text() == 'Expand':
            wtree.process_expand()
            self.but_exp_col.setIcon(icon.icon_folder_closed)
            but.setText('Collapse')
        else:
            wtree.process_collapse()
            self.but_exp_col.setIcon(icon.icon_folder_open)
            but.setText('Expand')


#    def on_but_clicked(self):
#        for but in self.list_of_buts:
#            if but.hasFocus(): break
#        logger.info('Click on "%s"' % but.text())
#        if   but == self.but_exp_col : self.expand_collapse_dbtree()
#        elif but == self.but_tabs    : self.view_hide_tabs()
#        elif but == self.but_buts    : self.select_visible_buttons()
#        elif but == self.but_del     : self.delete_selected_items()
#        elif but == self.but_docs    : self.select_doc_widget()
#        elif but == self.but_selm    : self.set_selection_mode()
#        elif but == self.but_add     : self.add_selected_item()
#        elif but == self.but_save    : self.save_selected_item()
#        #elif but == self.but_level   : self.set_logger_level()


if __name__ == "__main__":
    import os
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1' #export LIBGL_ALWAYS_INDIRECT=1
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    #logger.setPrintBits(0o177777)
    w = IVControl()
    #w.setGeometry(200, 400, 500, 200)
    w.setWindowTitle('IV Control Panel')
    w.show()
    app.exec_()
    del w
    del app

# EOF
