
"""Class :py:class:`QWInfoPanel` is a QWidget for information panel
===================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/QWInfoPanel.py

    from psana.graphqt.QWInfoPanel import QWInfoPanel
    w = QWInfoPanel()
    w.append(s, fname='info.txt')

Created on 2021-08-12 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QTextEdit, QMessageBox
from PyQt5.QtGui import QTextCursor, QFont
from psana.graphqt.QWIcons import icon

FNAME_INFO_DEF = 'info.txt'

def confirm_or_cancel_dialog_box(parent=None, text='confirm or cancel', title='Confirm or cancel') :
    """Pop-up MODAL box for confirmation"""
    mesbox = QMessageBox(parent, windowTitle=title, text=text,
                         standardButtons=QMessageBox.Ok | QMessageBox.Cancel)
    mesbox.setDefaultButton(QMessageBox.Ok)
    return mesbox.exec_() == QMessageBox.Ok


def save_textfile(text, path, mode='w'):
    """Saves text in file path with mode 'w'-write or 'a'-append"""
    logger.debug('save file %s' % path)
    f=open(path, mode)
    f.write(text)
    f.close() 


class QWInfoPanel(QWidget):
    """QWidget for information panel"""

    def __init__(self, **kwa):

        QWidget.__init__(self, kwa.get('parent', None))

        self.fname = kwa.get('fname_info', FNAME_INFO_DEF)
        self.count_empty = 0

        self.winfo = QTextEdit('Info panel')
        self.but_save = QPushButton('Save')
        self.but_clear = QPushButton('Clear')
        self.but_save.clicked.connect(self.on_but_save)
        self.but_clear.clicked.connect(self.on_but_clear)

        self.hbox = QHBoxLayout()
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.but_save)
        self.hbox.addWidget(self.but_clear)

        self.vbox = QVBoxLayout()
        self.vbox.addLayout(self.hbox)
        self.vbox.addWidget(self.winfo)
        self.setLayout(self.vbox)

        self.set_tool_tips()
        self.set_style()


    def set_tool_tips(self):
        self.but_save.setToolTip('Save info panel content in file')
        self.but_clear.setToolTip('Clean info panel')
 

    def set_style(self):
        icon.set_icons()
        self.but_save.setIcon(icon.icon_save)
        self.but_save.setFixedWidth(60)
        self.but_clear.setFixedWidth(60)
        self.layout().setContentsMargins(0,0,0,0)
        self.hbox.layout().setContentsMargins(2,2,2,0)

        w = self.winfo
        w.setReadOnly(True)
        w.setFont(QFont('monospace'))
        #cursor = w.textCursor()
        #w.selectAll()
        #w.setFontPointSize(32)
        #w.setTextCursor(cursor)


    def set_info_filename(self, fname=FNAME_INFO_DEF):
        self.fname = fname


    def on_but_save(self):
        logger.debug('on_but_save')
        fname = self.fname
        s = self.winfo.toPlainText()
        resp = confirm_or_cancel_dialog_box(parent=self, text='save info in file: %s' % fname, title='Confirm or cancel')
        logger.debug('file saving is confirmed: %s' % str(resp))
        if resp: save_textfile(str(s), fname, mode='w')


    def on_but_clear(self):
        logger.debug('on_but_clear')
        self.winfo.clear()
        self.count_empty = 0


    def remove_last_line(self):
        """trick removing last line"""
        w = self.winfo
        curspos = w.textCursor()
        w.moveCursor(QTextCursor.End, QTextCursor.MoveAnchor)
        w.moveCursor(QTextCursor.StartOfLine, QTextCursor.MoveAnchor)
        w.moveCursor(QTextCursor.End, QTextCursor.KeepAnchor)
        w.textCursor().removeSelectedText()
        w.textCursor().deletePreviousChar()
        w.setTextCursor(curspos)


    def append(self, s, fname=FNAME_INFO_DEF):
        is_empty_record = not s
        if is_empty_record:
           self.count_empty += 1
           s='%s times buffer is empty - click on stop' % str(self.count_empty)

           if self.count_empty>5: self.remove_last_line()
           #self.winfo.insertPlainText(s)
        #else:
        self.winfo.append(s)
        #self.winfo.insertPlainText(s)
        self.winfo.moveCursor(QTextCursor.End)#, QTextCursor.KeepAnchor)
        self.winfo.repaint()
        self.fname = fname
        #self.raise_()


    if __name__ == "__main__":

      def test_append(self, nlines=5):
          s = '== test_append'
          for i in range(nlines): s += '\n  line %04d' % i
          self.append(s)


      def keyPressEvent(self, e):
        logger.info('keyPressEvent, key=%s' % e.key())
        from PyQt5.QtCore import Qt
        if   e.key() == Qt.Key_Escape: self.close()
        elif e.key() == Qt.Key_A: self.test_append()
        else: logger.info('Keys:'\
               '\n  ESC - exit'\
               '\n  A - append info'\
               '\n')


if __name__ == "__main__":
    import os
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'
    logging.basicConfig(format='[%(levelname).1s] %(name)s L%(lineno)04d: %(message)s', level=logging.DEBUG)

    app = QApplication([])
    w = QWInfoPanel()
    w.setGeometry(100, 50, 300, 200)
    w.setWindowTitle('QWInfoPanel')
    w.append('add some info')
    w.append('add more')
    w.append('and more')
    w.show()
    app.exec_()
    del w
    del app

# EOF
