
"""Class :py:class:`QWTextBrowser` is a QWidget for text browsing
=================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/QWTextBrowser.py

    from psana.graphqt.QWTextBrowser import QWTextBrowser
    w = QWTextBrowser(path=fname)
    w.on_changed_fname(fname)

Created on 2021-08-05 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)

from psana.graphqt.CMWControlBase import *
from psana.pyalgos.generic.Utils import load_textfile

class QWTextBrowser(CMWControlBase):
    """CMWControlBase/QWidget for text browser with control fields"""

    def __init__(self, **kwa):

        kwa.setdefault('parent', None)
        kwa.setdefault('path', '/cds/group/psdm/detector/data2_test/misc/Select') #test.txt'
        kwa.setdefault('label', 'File:')
        kwa.setdefault('fltr', '*.txt *.text *.dat *.data *.meta\n*')
        kwa.setdefault('dirs', dirs_to_search())

        last_selected_fname = cp.last_selected_fname.value()
        last_selected_data = cp.last_selected_data

        path = kwa['path']
        if os.path.basename(path)=='Select' and last_selected_fname is not None: path = last_selected_fname

        CMWControlBase.__init__(self, **kwa)

        self.is_editable = kwa.get('is_editable', True)

        self.edi_txt = QTextEdit()

        self.hbox2 = QHBoxLayout()
        self.hbox2.addWidget(self.edi_txt)
        self.hbox1 = QHBoxLayout()
        self.hbox1.addSpacing(5)
        self.hbox1.addWidget(self.wfnm)
        self.hbox1.addStretch(1)
        self.hbox1.addWidget(self.but_save)
        self.hbox1.addWidget(self.but_view)
        self.hbox1.addWidget(self.but_tabs)
        self.hbox1.addSpacing(5)
        self.vbox = QVBoxLayout()
        self.vbox.addSpacing(5)
        self.vbox.addLayout(self.hbox1)
        self.vbox.addLayout(self.hbox2)
        self.setLayout(self.vbox)

        self.wfnm.connect_path_is_changed(self.on_changed_fname)

        self.set_tool_tips()
        self.set_style()

        cp.qwtextbrowser = self

        if os.path.basename(path) != 'Select': self.on_changed_fname(path)
        elif last_selected_data: self.set_text(str(last_selected_data))
        else: self.set_text('File name and data are missing... Select the file name.')


    def set_tool_tips(self):
        CMWControlBase.set_tool_tips(self)
        #self.setToolTip('Text Browser')


    def set_style(self):
        CMWControlBase.set_style(self)
        self.setStyleSheet(style.styleBkgd)
        self.edi_txt.setReadOnly(not self.is_editable)
        self.edi_txt.setStyleSheet(style.styleWhiteFixed)
        self.layout().setContentsMargins(0,0,0,0)
        #self.wfnm.layout().setContentsMargins(5,0,0,0)


    def set_text(self, txt):
        self.edi_txt.setText(txt)


    def on_changed_fname(self, fname):
        logger.debug('on_changed_fname: %s' % fname)
        txt = load_textfile(fname)
        logger.debug('loaded %d lines, %d chars from file %s' % (txt.count('\n'), len(txt), fname))

        txtbut = str(self.wfnm.but.text())
        if str(fname) != txtbut: self.wfnm.but.setText(str(fname))

        self.set_text(self, txt)(txt)


    def closeEvent(self, e):
        logger.debug('closeEvent')
        CMWControlBase.closeEvent(self, e)
        cp.qwtextbrowser = None


    def on_but_view(self):
        logger.debug('on_but_view should be re-implemented')


    def on_but_save(self):
        logger.debug('re-implemented on_but_save')
        from psana.graphqt.QWUtils import get_save_fname_through_dialog_box
        fname = 'tmp.txt'
        path = get_save_fname_through_dialog_box(self, fname, 'Select file to save', filter='*.txt')
        if path is None or path == '': return
        text = str(self.edi_txt.toPlainText())
        logger.info('Save in file:\n%s' % text)
        f=open(path,'w')
        f.write(text)
        f.close()


if __name__ == "__main__":
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1' #export LIBGL_ALWAYS_INDIRECT=1
    logging.basicConfig(format='[%(levelname).1s] L:%(lineno)03d %(name)s %(message)s', level=logging.DEBUG)

    app = QApplication(sys.argv)
    w = QWTextBrowser()
    w.setGeometry(100, 50, 900, 500)
    w.setWindowTitle('Text Browser')
    w.show()
    app.exec_()

# EOF
