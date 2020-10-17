#--------------------
"""
:py:class:`CGWConfigEditor` - widget for configuration editor
============================================================================================

Usage::

    # Import
    from psdaq.control_gui.CGWConfigEditor import CGWConfigEditor

    # Methods - see test

See:
    - :py:class:`CGWConfigEditor`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2/psdaq/psdaq/control_gui>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2019-03-08 by Mikhail Dubrovin
"""
#--------------------

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QTabBar
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,\
                            QPushButton, QFileDialog, QComboBox

from psdaq.control_gui.CGWConfigEditorTree import CGWConfigEditorTree
from psdaq.control_gui.CGWConfigEditorText import CGWConfigEditorText
from psdaq.control_gui.Utils import save_textfile, path_to_test_data
from psdaq.control_gui.CGJsonUtils import load_json_from_file, str_json, json_from_str

from psdaq.control_gui.QWUtils import help_dialog_box
from psdaq.control_gui.CGConfigParameters import cp

#--------------------

char_expand  = u' \u25BC' # down-head triangle
char_shrink  = u' \u25B2' # solid up-head triangle

#--------------------

class CGWConfigEditor(QWidget):
    """Configuration editor top widget
    """
    MORE_OPTIONS = ('More','Load','Save','Help')
    EDITOR_TYPES = ('Text','Tree')

    def __init__(self, parent=None, dictj=None):

        #QGroupBox.__init__(self, 'Partition', parent)
        QWidget.__init__(self, parent)

        cp.cgwconfigeditor = self

        # I/O files
        self.ifname_json = '%s/json2xtc_test.json' % path_to_test_data() # input file
        self.ofname_json = './test.json' # output file

        self.help_box = None

        self.dictj = dictj
        if dictj is None: self.load_dict() # fills self.dictj

        self.but_close= QPushButton('Close')
        self.but_apply= QPushButton('Apply')
        self.but_expn = QPushButton('Expand %s'%char_expand)
        self.box_type = QComboBox(self)

        self.box_type.addItems(self.EDITOR_TYPES)
        self.box_type.setCurrentIndex(1)
        self.wedi = CGWConfigEditorTree(dictj=self.dictj)

        self.box_more = QComboBox(self)
        self.box_more.addItems(self.MORE_OPTIONS)
        self.box_more.setCurrentIndex(0)

        self.hbox = QHBoxLayout() 
        self.hbox.addWidget(self.but_apply)
        self.hbox.addWidget(self.but_close)
        self.hbox.addWidget(self.but_expn)
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.box_more)
        self.hbox.addWidget(self.box_type)

        self.vbox = QVBoxLayout() 
        self.vbox.addLayout(self.hbox)
        self.vbox.addWidget(self.wedi)
        self.setLayout(self.vbox)

        self.set_tool_tips()
        self.set_style()

        self.but_expn.clicked.connect(self.on_but_expn)
        self.but_apply.clicked.connect(self.on_but_apply)
        self.but_close.clicked.connect(self.on_but_close)
        self.box_type.currentIndexChanged[int].connect(self.on_box_type)
        self.box_more.currentIndexChanged[int].connect(self.on_box_more)

#--------------------

    def set_tool_tips(self):
        self.setToolTip('Configuration editor GUI'\
                        '\n - Expand/Collapse button - expands/collapse entire dictionary content'\
                        '\n - Close button - closes editor window, all modifications'\
                        '     since last click on Apply will be lost'\
                        '\n - Tree/Text button - switches between editor in "tree" and "text" modes'\
                        '\n - More button - selects other editor options'\
                        '\n - click on triangle -  expands/collapses particular item'\
                        '\n - * (star) to the right of item title indicates that item is compound object'\
                        '\n - double-click on value starts in-line editor or compound item editor window'\
                        '\n - click on rectangular box in front of INT value swap presentation between'\
                        '\n   DEC/HEX/OCT/BIN'\
                       )
        self.box_type.setToolTip('Select editor type')
        self.but_expn.setToolTip('Expand/Collapse tree-like content')
        self.but_apply.setToolTip('Apply content changes to configuration DB')
        self.but_close.setToolTip('Close window, cancel all modifications since last click on Apply button')
        self.box_more.setToolTip('More options:'\
                                +'\n * Load content from file'\
                                +'\n * Save content in file')

#--------------------

    def set_style(self):
        from psdaq.control_gui.Styles import style
        self.setWindowTitle('Configuration Editor')
        self.setMinimumSize(400,800)
        self.layout().setContentsMargins(0,0,0,0)
  
        self.but_apply.setStyleSheet(style.styleButtonGood) 
        #self.but_save.setVisible(True)
 
        #self.setMinimumWidth(300)
        #self.edi.setMinimumWidth(210)
        #self.setFixedHeight(34) # 50 if self.show_frame else 34)
        #if not self.show_frame:
        #self.layout().setContentsMargins(0,0,0,0)

        #style = "background-color: rgb(255, 255, 220); color: rgb(0, 0, 0);" # Yellowish
        #style = "background-color: rgb(100, 240, 200); color: rgb(0, 0, 0);" # Greenish
        #style = "background-color: rgb(255, 200, 220); color: rgb(0, 0, 0);" # Pinkish
        #style = "background-color: rgb(240, 240, 100); color: rgb(0, 0, 0);" # YellowBkg
        #self.setStyleSheet(style)

        #self.setFixedSize(750,270)
        #self.setMaximumWidth(800)
 
#--------------------

    def load_dict(self):
        ifname = self.ifname_json
        logger.info('CGWConfigEditor: load json from %s' % ifname)
        self.dictj = dj = load_json_from_file(ifname)
        sj = str_json(dj)
        logger.debug('CGWConfigEditor: dict of json as str:\n%s\nconfiguration object type: %s' % (sj, type(dj)))

#--------------------
 
    def on_but_load(self):
        logger.debug('on_but_load')
        if self.select_ifname():
           self.load_dict()
           self.wedi.set_content(self.dictj)

#--------------------
 
    def select_ifname(self):
        logger.info('select_ifname %s' % self.ifname_json)
        path, ftype = QFileDialog.getOpenFileName(self,
                      caption   = 'Select the file to load json',
                      directory = self.ifname_json,
                      filter    = 'Text files(*.json *.cfg *.txt *.text *.dat *.data)\nAll files (*)'
                      )
        if path == '':
            logger.info('Loading is cancelled')
            return False
        self.ifname_json = path
        logger.info('Input file: %s' % path)
        return True

#--------------------
 
    def on_but_save(self):
        logger.debug('on_but_save')
        if self.select_ofname():
           self.save_dict_in_file()

#--------------------
 
    def on_but_help(self):
        logger.debug('on_but_help')
        s = self.toolTip()
        if self.help_box is not None: self.help_box.close()
        self.help_box = help_dialog_box(parent=self.box_more,
                                        text=self.toolTip(), title='Help')

#--------------------
 
    def select_ofname(self):
        logger.info('select_ofname %s' % self.ofname_json)
        path, ftype = QFileDialog.getSaveFileName(self,
                      caption   = 'Select the file to save json',
                      directory = self.ofname_json,
                      filter    = 'Text files (*.json *.cfg *.txt *.text *.dat *.data)\nAll files (*)'
                      )
        if path == '':
            logger.info('Saving is cancelled')
            return False
        self.ofname_json = path
        logger.info('Output file: %s' % path)
        return True

#--------------------

    def get_content(self):
        self.dictj = self.wedi.get_content()
        return self.dictj

#--------------------
 
    def save_dict_in_file(self):
        logger.info('save_dict_in_file %s' % self.ofname_json)
        dj = self.get_content()
        sj = str_json(dj)
        save_textfile(sj, self.ofname_json, mode='w', verb=True)

#--------------------
 
    def on_but_apply(self):
        logger.debug('on_but_apply')
        dj = self.get_content()
        sj = str_json(dj)
        logger.info('on_but_apply jason/dict:\n%s' % sj)

        if cp.cgwmainconfiguration is None:
            logger.warning("parent (ctrl) is None - changes can't be applied to DB")
            return
        else:
            cp.cgwmainconfiguration.save_dictj_in_db(dj, msg='CGWConfigEditor: ')

#--------------------
 
    def on_but_close(self):
        logger.debug('on_but_close')
        self.close()

#--------------------
 
    def on_box_more(self, ind):
        opt = self.MORE_OPTIONS[ind]
        logger.info('CGWConfigEditor selected option %s' % opt)

        if   ind==1: self.on_but_load()
        elif ind==2: self.on_but_save()
        elif ind==3: self.on_but_help()
        self.box_more.setCurrentIndex(0)

#--------------------
 
    def on_box_type(self, ind):
        type = self.EDITOR_TYPES[ind]
        logger.info('CGWConfigEditor set editor type %s' % type)

        if self.wedi is not None:
           self.vbox.removeWidget(self.wedi)
           self.wedi.close()
           del self.wedi

        self.wedi = self.editor_widget_selector(type)
        self.vbox.addWidget(self.wedi)
        self.wedi.setVisible(True)

        #self.show_documents(self.dbname, self.colname)

#--------------------

    def editor_widget_selector(self, edi_type):
        """Factory method for selection of the editor widget.
        """
        logger.info('Set document browser in mode %s' % edi_type)

        is_tree_editor = edi_type=='Tree'
        self.but_expn.setVisible(is_tree_editor)
        if is_tree_editor: self.but_expn.setText('Expand %s'%char_expand)

        kwargs = {'parent':None, 'dictj':self.dictj}

        if   edi_type == self.EDITOR_TYPES[0]: return CGWConfigEditorText(**kwargs)
        elif edi_type == self.EDITOR_TYPES[1]: return CGWConfigEditorTree(**kwargs)
        else:
            logger.warning('Unknown editor type "%s"' % edi_type)
            return QTextEdit(edi_type)

#--------------------
 
    def on_but_expn(self):
        #logger.debug('on_but_expn')
        if self.but_expn.text()[:6] == 'Expand':
           self.wedi.process_expand()
           self.but_expn.setText('Collapse %s'%char_shrink)
        else:
           self.wedi.process_collapse()
           self.but_expn.setText('Expand %s'%char_expand)

#--------------------

    def closeEvent(self, e):
        QWidget.closeEvent(self, e)
        logger.debug('closeEvent')

        if self.help_box is not None: self.help_box.close()
        if cp.cgwconfigeditor is not None:
           cp.cgwconfigeditor = None

#--------------------

if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)

    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = CGWConfigEditor(None)
    w.show()
    app.exec_()

#--------------------
