#------------------------------
"""Class :py:class:`CMWDBDocs` is a QWidget for configuration parameters
==============================================================================

Usage ::
    # Test: python lcls2/psana/psana/graphqt/CMWDBDocs.py

    # Import
    from psana.graphqt.CMWDBDocs import CMWDBDocs

    # See test at the EOF

See:
  - :class:`CMWMain`
  - :class:`CMWDBDocs`
  - `on github <https://github.com/slac-lcls/lcls2>`_.

Created on 2017-04-05 by Mikhail Dubrovin
"""
#------------------------------

from psana.graphqt.CMConfigParameters import cp
from psana.pyalgos.generic.Logger import logger
from psana.graphqt.Styles import style

#from psana.graphqt.CMWDBDocsText  import CMWDBDocsText 
#from psana.graphqt.CMWDBDocsList  import CMWDBDocsList
#from psana.graphqt.CMWDBDocsTable import CMWDBDocsTable
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QTextEdit #, QTabBar, QLabel, QPushButton, QHBoxLayout, 
#from PyQt5.QtGui import QColor#, QFont
#from PyQt5.QtCore import Qt

#------------------------------

class CMWDBDocs(QWidget) :
    """CMWDBDocs is a QWidget with tabs for configuration management"""

    def __init__(self, parent=None) :
        QWidget.__init__(self, parent)
        self._name = 'CMWDBDocs'

        cp.cmwdbdocs = self

        #self.but_close  = QPushButton('&Close') 

        self.list_of_doc_widgets = cp.list_of_doc_widgets # ('Text','List','Table')

        self.hboxw = QHBoxLayout()
        self.gui_win = None
        self.gui_selector(cp.cdb_docw.value())

        self.vbox = QVBoxLayout()
        self.vbox.addLayout(self.hboxw)
        #self.vbox.addStretch(1)     
        #self.vbox.addLayout(self.hboxB)
        self.setLayout(self.vbox)

        #self.but_close.clicked.connect(self.on_close)

        self.set_tool_tips()
        self.set_style()


    def set_tool_tips(self):
        pass
        #self.but_close.setToolTip('Close this window.')


    def set_style(self):
        self.          setStyleSheet(style.styleBkgd)
        #self.but_close.setStyleSheet(style.styleButton)
        #self.setMinimumSize(600,360)
        self.setContentsMargins(-9,-9,-9,-9)


    def gui_selector(self, docw=None):

        if self.gui_win is not None : 
            self.gui_win.close()
            del self.gui_win

        w_height = 500

        docw_type = docw if docw is not None else cp.cdb_docw.value()

        print('gui_selector docw_type: %s' % docw_type)


        if docw_type == self.list_of_doc_widgets[0] :
            self.gui_win = QTextEdit(docw_type)

        elif docw_type == self.list_of_doc_widgets[1] :
            self.gui_win = QTextEdit(docw_type)
            w_height = 170

        elif docw_type == self.list_of_doc_widgets[2] :
            self.gui_win = QTextEdit(docw_type)

        else :
            logger.warning('Unknown doc widget name "%s"' % docw_type, self._name)

        #self.set_status(0, 'Set configuration file')
        #self.gui_win.setFixedHeight(w_height)
        self.hboxw.addWidget(self.gui_win)
        self.gui_win.setVisible(True)


    def show_documents(self, dbname, colname) :

        #self.gui_win.show_documents(dbname, colname)
        # use temporary solution

        import psana.graphqt.CMDBUtils as dbu

        txt = dbu.collection_info(dbname, colname)
        self.gui_win.setText(txt)


    #def resizeEvent(self, e):
        #logger.debug('resizeEvent', self._name) 
        #print self._name + ' config: self.size():', self.size()
        #self.setMinimumSize( self.size().width(), self.size().height()-40 )
        #pass


    #def moveEvent(self, e):
        #logger.debug('moveEvent', self._name) 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #logger.debug('moveEvent: new pos:' + str(self.position), self._name)
        #pass


    def closeEvent(self, e):
        logger.debug('closeEvent', self._name)
        #self.tab_bar.close()        
        if self.gui_win is not None : self.gui_win.close()
        QWidget.close(self)


    def key_usage(self) :
        return 'Keys:'\
               '\n  ESC - exit'\
               '\n  0 - set widget'\
               '\n  1 - set another widget'\
               '\n  2 - set another widget'\
               '\n'


    def keyPressEvent(self, e) :
        print('keyPressEvent, key=', e.key())       

        if   e.key() == Qt.Key_Escape :
            self.close()
        elif e.key() in (Qt.Key_0, Qt.Key_1, Qt.Key_2)  : 
            docw_type = self.list_of_doc_widgets[int(e.key())]
            cp.cdb_docw.setValue(docw_type)
            self.gui_selector()
        else :
            print(self.key_usage())

#-----------------------------

if __name__ == "__main__" :
    from PyQt5.QtWidgets import QApplication
    import sys
    logger.setPrintBits(0o177777)
    app = QApplication(sys.argv)
    w = CMWDBDocs()
    #w.setGeometry(1, 1, 600, 200)
    w.setWindowTitle('Document widge selector')
    w.show()
    app.exec_()

    #cp.printParameters()
    #cp.saveParametersInFile()

    del w
    del app

#-----------------------------
