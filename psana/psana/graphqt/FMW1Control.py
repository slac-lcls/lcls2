
"""Class :py:class:`FMW1Control` is a File Manager QWidget with control fields
==============================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/FMW1Control.py

    from psana.graphqt.FMW1Control import FMW1Control
    w = FMW1Control()

Created on 2021-07-27 by Mikhail Dubrovin
"""
import os

import logging
logger = logging.getLogger(__name__)

from psana.graphqt.CMWControlBase import CMWControlBase
from PyQt5.QtWidgets import QApplication, QGridLayout, QPushButton
from psana.graphqt.CMConfigParameters import cp, dir_calib
from psana.graphqt.QWUtils import change_check_box_dict_in_popup_menu
from PyQt5.QtCore import pyqtSignal, QModelIndex

from psana.graphqt.QWIcons import icon


class FMW1Control(CMWControlBase):
    """QWidget for File Manager control fields"""

    instr_exp_is_changed = pyqtSignal('QString', 'QString')

    def __init__(self, **kwargs):

        parent = kwargs.get('parent', None)

        CMWControlBase.__init__(self, **kwargs)
        cp.fmw1control = self

        self.but_exp     = QPushButton(cp.exp_name.value())
        self.but_exp_col = QPushButton('Collapse')
        self.but_save    = QPushButton('Save')

        self.box = QGridLayout()
        self.box.addWidget(self.but_exp,     0, 0)
        self.box.addWidget(self.but_exp_col, 0, 1)
        self.box.addWidget(self.but_save,    0, 2)
        self.box.addWidget(self.but_tabs,    0, 9)
        self.setLayout(self.box)
 
        self.but_exp.clicked.connect(self.on_but_exp)
        self.but_exp_col.clicked.connect(self.on_but_exp_col)
        self.but_save.clicked.connect(self.on_but_save)

#        self.connect_instr_exp_is_changed(self.on_instr_exp_is_changed)

        self.set_tool_tips()
        self.set_style()

        if cp.fmw1main is not None:
            global full_path_for_item
            from psana.graphqt.FSTree import full_path_for_item
            cp.fmw1main.wfstree.connect_item_selected_to(self.on_item_selected)
            cp.fmw1main.wfstree.clicked[QModelIndex].connect(self.on_click)


    def set_tool_tips(self):
        CMWControlBase.set_tool_tips(self)
        self.but_tabs.setToolTip('Show/hide tabs')
        self.but_exp.setToolTip('Select experiment')
        self.but_save.setToolTip('To save array in file\nclick on *.data file in the tree\nthen click on this Save button')
        self.but_exp_col.setToolTip('Collapse/expand directory/file tree')


    def set_style(self):
        icon.set_icons()
        self.but_exp_col.setIcon(icon.icon_folder_open)
        self.but_save.setIcon(icon.icon_save)
        self.but_tabs.setFixedWidth(50)
        self.but_exp.setFixedWidth(80)
        self.but_exp_col.setFixedWidth(80)
        self.but_save.setFixedWidth(80)
         #self.but_buts.setStyleSheet(style.styleButton)
        #self.but_tabs.setVisible(True)


    def on_but_exp(self):
        from psana.graphqt.PSPopupSelectExp import select_instrument_experiment
        dir_instr = cp.instr_dir.value()
        instr_name, exp_name = select_instrument_experiment(self.but_exp, dir_instr, show_frame=True)
        logger.debug('selected experiment: %s' % exp_name)
        if instr_name and exp_name and exp_name!=cp.exp_name.value():
            self.but_exp.setText(exp_name)
            cp.instr_name.setValue(instr_name)
            cp.exp_name.setValue(exp_name)

            if cp.fmw1main is not None:
               cp.fmw1main.wfstree.update_tree_model(dir_calib())


    def on_but_exp_col(self):
        logger.debug('on_but_exp_col')
        if cp.fmw1main is None: return
        wtree = cp.fmw1main.wfstree
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
        logger.debug('TBD on_but_save')
        if cp.fmw1main is None: return
        wtree = cp.fmw1main.wfstree

        names_selected = [full_path_for_item(i) for i in wtree.selected_items()]
        logger.info('on_but_save selected names:\n  %s' % '\n  '.join(names_selected))

        return


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
        else: 
            logger.info('command "Save" is cancelled')


#            self.instr_exp_is_changed.emit(instr_name, exp_name)

#    def connect_instr_exp_is_changed(self, recip):
#        self.instr_exp_is_changed.connect(recip)


#    def disconnect_instr_exp_is_changed(self, recip):
#        self.instr_exp_is_changed.disconnect(recip)


#    def on_instr_exp_is_changed(self, instr, exp):
#        logger.debug('selected instrument: %s experiment: %s' % (instr, exp))


    def view_hide_tabs(self):
        CMWControlBase.view_hide_tabs(self)
        wtabs = cp.fmwtabs
        if wtabs is None: return
        is_visible = wtabs.tab_bar_is_visible()
        #self.but_tabs.setText('Tabs %s'%cp.char_shrink if is_visible else 'Tabs %s'%cp.char_expand)
        wtabs.set_tabs_visible(not is_visible)


    def save_item_path(self, index):
        i = cp.fmw1main.wfstree.model.itemFromIndex(index)
        if i is None:
           logger.debug('on_item_selected: item is None')
           return
        fname = full_path_for_item(i)
        if os.path.exists(fname) and (not os.path.isdir(fname)):
            logger.info('save_item_path save last_selected_fname: %s path: %s' % (str(i.text()), fname))
            cp.last_selected_fname.setValue(fname)


    def on_item_selected(self, selected, deselected):
        self.save_item_path(selected)


    def on_click(self, index):
        self.save_item_path(index)


    def closeEvent(self, e):
        logger.debug('closeEvent')
        CMWControlBase.closeEvent(self, e)
        cp.fmw1control = None


if __name__ == "__main__":
    logging.basicConfig(format='[%(levelname).1s] %(name)s L%(lineno)04d : %(message)s', level=logging.DEBUG)
    import sys
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1' #export LIBGL_ALWAYS_INDIRECT=1

    app = QApplication(sys.argv)
    w = FMW1Control()
    w.setGeometry(100, 50, 500, 80)
    w.setWindowTitle('FM Control Panel')
    w.show()
    app.exec_()
    del w
    del app

# EOF
