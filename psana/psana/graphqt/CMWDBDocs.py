
"""Class :py:class:`CMWDBDocs` is a QWidget for configuration parameters
========================================================================

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

import logging
logger = logging.getLogger(__name__)

from psana.graphqt.CMConfigParameters import cp
from psana.graphqt.Styles import style

from psana.graphqt.CMDBUtils import dbu
from psana.graphqt.CMWDBDocsText  import CMWDBDocsText
from psana.graphqt.CMWDBDocsList  import CMWDBDocsList
from psana.graphqt.CMWDBDocsTable import CMWDBDocsTable
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QTextEdit

def docs_widget_selector(dwtype):
    """Factory method for selection of the document widget.
    """
    dwtypes = cp.list_of_doc_widgets

    logger.info('Set document browser in mode %s' % dwtype)

    if   dwtype == dwtypes[0]: return CMWDBDocsText()
    elif dwtype == dwtypes[1]: return CMWDBDocsList()
    elif dwtype == dwtypes[2]: return CMWDBDocsTable()
    else:
        logger.warning('Unknown doc widget type "%s"' % dwtype)
        return QTextEdit(dwtype)


class CMWDBDocs(QWidget):
    """CMWDBDocs is a QWidget with tabs for configuration management"""

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self._name = 'CMWDBDocs'

        cp.cmwdbdocs = self

        self.dbname  = None
        self.colname = None

        self.list_of_doc_widgets = cp.list_of_doc_widgets

        self.hboxw = QHBoxLayout()
        self.gui_win = None
        self.set_docs_widget(cp.cdb_docw.value())

        self.vbox = QVBoxLayout()
        self.vbox.addLayout(self.hboxw)
        self.setLayout(self.vbox)

        self.set_tool_tips()
        self.set_style()


    def set_tool_tips(self):
        pass


    def set_style(self):
        self.setStyleSheet(style.styleBkgd)
        self.layout().setContentsMargins(0,0,0,0)


    def set_docs_widget(self, docw=None):

        if self.gui_win is not None:
            self.gui_win.close()
            del self.gui_win

        docw_type = docw if docw is not None else cp.cdb_docw.value()
        self.gui_win = docs_widget_selector(docw_type)
        self.hboxw.addWidget(self.gui_win)
        self.gui_win.setVisible(True)

        self.show_documents(self.dbname, self.colname)


    def show_documents(self, dbname, colname, force_update=False):

        if None in (dbname, colname): return

        if ((dbname, colname) != (self.dbname, self.colname))\
        or force_update:
            self.current_docs = dbu.list_of_documents(dbname, colname)
            self.dbname, self.colname = dbname, colname

        self.gui_win.show_documents(dbname, colname, self.current_docs)


    def closeEvent(self, e):
        logger.debug('closeEvent')
        if self.gui_win is not None: self.gui_win.close()
        QWidget.close(self)

    if __name__ == "__main__":

      def key_usage(self):
        return 'Keys:'\
               '\n  ESC - exit'\
               '\n  0 - set widget'\
               '\n  1 - set another widget'\
               '\n  2 - set another widget'\
               '\n'


      def keyPressEvent(self, e):
        logger.info('keyPressEvent, key=', e.key())

        if   e.key() == Qt.Key_Escape:
            self.close()
        elif e.key() in (Qt.Key_0, Qt.Key_1, Qt.Key_2):
            docw_type = self.list_of_doc_widgets[int(e.key())]
            cp.cdb_docw.setValue(docw_type)
            self.set_docs_widget()
        else:
            logger.info(self.key_usage())


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)
    app = QApplication(sys.argv)
    w = CMWDBDocs()
    #w.setGeometry(1, 1, 600, 200)
    w.setWindowTitle('Document widge selector')
    w.show()
    app.exec_()
    del w
    del app

# EOF
