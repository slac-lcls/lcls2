
"""Class :py:class:`H5VControl` is a QWidget with control fields
===================================================================
Usage ::

    # Run test: python lcls2/psana/psana/graphqt/H5VControl.py

    from psana.graphqt.H5VControl import H5VControl
    w = H5VControl()

Created on 2020-01-04 by Mikhail Dubrovin
"""
import logging
logger = logging.getLogger(__name__)

import sys

from psana.graphqt.CMWControlBase import CMWControlBase
from PyQt5.QtWidgets import QApplication, QLabel, QComboBox, QPushButton, QHBoxLayout, QLineEdit
from PyQt5.QtCore import QSize
from psana.graphqt.CMConfigParameters import cp, dirs_to_search
from psana.graphqt.Styles import style
from psana.graphqt.QWIcons import icon
from psana.graphqt.QWFileNameV2 import QWFileNameV2

from psana.pyalgos.generic.NDArrUtils import info_ndarr, np

def save_data_in_file(data, prefix, control={'txt': True, 'npy': True}, fmt='%.3f'):
    #elif data_type == 'any':
    #    gu.save_textfile(str(data), fname, mode='w', verb=verb)
    if isinstance(data, np.ndarray):
        if control['txt']:
            from psana.pscalib.calib.NDArrIO import save_txt # load_txt
            fname = '%s.txt' % prefix
            save_txt(fname, data, cmts=(), fmt=fmt)
        if control['npy']:
            fname = '%s.npy' % prefix
            np.save(fname, data, allow_pickle=False)
            logger.info('saved file: %s' % fname)
    else:
        logger.warning('DO NOT SAVE unexpected data type: %s' % type(data))


class H5VControl(CMWControlBase):
    """CMWControlBase for H5V control fields"""

    def __init__(self, **kwargs):

        parent = kwargs.get('parent', None)
        CMWControlBase.__init__(self, **kwargs)
        self.lab_ctrl = QLabel('Control:')

        self.but_exp_col = QPushButton('Collapse')
        self.but_save    = QPushButton('Save')

        fname = cp.h5vmain.wtree.fname if cp.h5vmain is not None else './test.h5'

        self.w_fname = QWFileNameV2(None, label='HDF5 file:',\
           path=fname, fltr='*.h5 *.hdf5 \n*', dirs=dirs_to_search())

        self.hbox = QHBoxLayout() 
        self.hbox.addWidget(self.w_fname)
        self.hbox.addWidget(self.lab_ctrl)
        self.hbox.addWidget(self.but_exp_col)
        self.hbox.addStretch(1) 
        self.hbox.addWidget(self.but_save)
        self.hbox.addWidget(self.but_tabs)
        #self.hbox.addSpacing(20)
        self.setLayout(self.hbox)
 
        self.but_exp_col.clicked.connect(self.on_but_exp_col)
        self.but_save.clicked.connect(self.on_but_save)

        if cp.h5vmain is not None:
            self.w_fname.connect_path_is_changed_to_recipient(cp.h5vmain.wtree.set_file)
            cp.h5vmain.wtree.connect_item_selected_to(self.on_item_selected)

        self.set_tool_tips()
        self.set_style()
        #self.set_buttons_visiable()


    def set_tool_tips(self):
        CMWControlBase.set_tool_tips(self)
        #self.but_tabs.setToolTip('Show/hide tabs')
        #self.setToolTip('Control fields/buttons')
        self.but_save.setToolTip('To save array in file\nclick on numpy.array in the tree\nthen click on this Save button')
        self.but_exp_col.setToolTip('Collapse/expand hdf5 tree')
        self.w_fname.but.setToolTip('hdf5 file selection')


    def set_style(self):
        self.lab_ctrl.setStyleSheet(style.styleLabel)
        self.w_fname.lab.setStyleSheet(style.styleLabel)
        icon.set_icons()
        self.but_exp_col.setIcon(icon.icon_folder_open)
        self.but_save.setIcon(icon.icon_save)
        #self.but_save.setStyleSheet('') #style.styleButton, style.styleButtonGood
        self.enable_but_save()
        self.but_tabs.setFixedWidth(50)


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


    def on_but_save(self):
        logger.debug('on_but_save')

        prefix = self.dname
        data = self.data
        logger.info(info_ndarr(data, 'data:'))
        logger.info('data name prefix: %s' % prefix)

        from psana.graphqt.QWUtils import change_check_box_dict_in_popup_menu

        control = {'txt': True, 'npy': True}
        resp = change_check_box_dict_in_popup_menu(control, 'Select and confirm',\
                 msg='save selected data in file\n%s\nfor types:'%prefix, parent=None) #self.but_save)

        if resp==1:
            logger.info('save hdf5 data in file(s) with prefix: %s' % prefix)
            fmt = '%d' if 'int' in str(data.dtype) else '%.3f'
            save_data_in_file(data, prefix, control, fmt)
            cp.last_selected_fname.setValue('%s.npy' % prefix)
        else: 
            logger.info('command "Save" is cancelled')


    def enable_but_save(self, is_good=False):
        self.but_save.setStyleSheet(style.styleButtonGood if is_good else style.styleButton)
        self.but_save.setFlat(not is_good)
        self.but_save.setEnabled(is_good)
        if not is_good:
          self.dname = None
          self.data = None


    def on_item_selected(self, selected, deselected):
        logger.debug('TBD on_item_selected')
        wtree = cp.h5vmain.wtree
        self.enable_but_save()
        itemsel = wtree.model.itemFromIndex(selected)
        if itemsel is None: return
        if isinstance(itemsel.data(), wtree.h5py.Dataset):
            logger.debug('data.value:\n%s' % str(itemsel.data()[()]))

            data = self.data = itemsel.data()[()]
            dname = self.dname = wtree.full_path(itemsel)
            logger.info(info_ndarr(data, 'data:'))
            logger.info('full name: %s' % dname)

            is_good_to_save = isinstance(data, np.ndarray) and data.size>1
            self.enable_but_save(is_good_to_save)


#    def on_but_clicked(self):
#        for but in self.list_of_buts:
#            if but.hasFocus(): break
#        logger.info('Click on "%s"' % but.text())
#        if   but == self.but_exp_col : self.expand_collapse_dbtree()
#        elif but == self.but_tabs    : self.view_hide_tabs()
#        #elif but == self.but_level   : self.set_logger_level()


if __name__ == "__main__":

    import os
    import sys
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'

    app = QApplication(sys.argv)
    w = H5VControl()
    #w.setGeometry(200, 400, 500, 200)
    w.setWindowTitle('H5V Control Panel')
    w.show()
    app.exec_()
    del w
    del app

# EOF
