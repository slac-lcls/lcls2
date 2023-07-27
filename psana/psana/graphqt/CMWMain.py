
"""
Class :py:class:`CMWMain` is a QWidget for interactive image
============================================================

Usage ::
    see calibman

See:
    - :class:`CMWMain`
    - :class:`CMWMainTabs`
    - :class:`CMConfigParameters`
    - `graphqt documentation <https://lcls-psana.github.io/graphqt/py-modindex.html>`_.

Created on 2017-02-01 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-26 by Mikhail Dubrovin
"""
import sys
import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QSplitter, QTextEdit
from PyQt5.QtCore import Qt, QPoint

from psana.detector.RepoManager import RepoManager
from psana.detector.Utils import gu
from psana.graphqt.CMConfigParameters import cp
from psana.graphqt.QWLoggerStd import QWLoggerStd

print_kwargs, is_in_command_line = gu.print_kwargs, gu.is_in_command_line
SCRNAME = sys.argv[0].rsplit('/')[-1]

class CMWMain(QWidget):

    def __init__(self, **kwargs):
        QWidget.__init__(self, parent=None, flags=Qt.WindowStaysOnTopHint)

        cp.cmwmain = self

        repoman = RepoManager(dirrepo=kwargs['repodir'])  # dettype=SCRNAME
        logfname = repoman.logname(SCRNAME)
        #print('log file name: %s' % logfname)

        self.set_input_pars(**kwargs)

        from psana.graphqt.CMWMainTabs import CMWMainTabs # AFTER set_input_pars !!!!\
        self.wlog = cp.wlog = QWLoggerStd(cp, show_buttons=False, logfname=logfname)

        repoman.save_record_at_start(SCRNAME)

        self.wtab = CMWMainTabs()

        self.vspl = QSplitter(Qt.Vertical)
        self.vspl.addWidget(self.wtab)
        self.vspl.addWidget(self.wlog)

        self.mbox = QHBoxLayout()
        self.mbox.addWidget(self.vspl)

        self.setLayout(self.mbox)

        self.set_style()
        #self.set_tool_tips()


    def set_input_pars(self, **kwa):

        cp.kwargs = kwa

        self.xywh = cp.main_win_pos_x .value(),\
                    cp.main_win_pos_y .value(),\
                    cp.main_win_width .value(),\
                    cp.main_win_height.value()

        x,y,w,h = self.xywh
        self.setGeometry(x,y,w,h)
        logger.info('set preserved window geometry x,y,w,h: %d,%d,%d,%d' % (x,y,w,h))
        #logger.info(log_rec_at_start())

        self.main_win_pos_x  = cp.main_win_pos_x
        self.main_win_pos_y  = cp.main_win_pos_y
        self.main_win_width  = cp.main_win_width
        self.main_win_height = cp.main_win_height

        host       = kwa.get('host', None)
        port       = kwa.get('port', None)
        cp.user    = kwa.get('user', None)
        cp.upwd    = kwa.get('upwd', None)
        exp        = kwa.get('experiment', None)
        det        = kwa.get('detector', None)
        repodir    = kwa.get('repodir', None)
        loglevel   = kwa.get('loglevel', 'DEBUG').upper()
        savecfg    = kwa.get('savecfg', False)
        savelog    = kwa.get('savelog', False)
        if isinstance(loglevel,str): loglevel = loglevel.upper()

        cp.save_log_at_exit.setValue(savelog)
        cp.log_level.setValue(loglevel)
        cp.log_prefix.setValue(repodir)

        if is_in_command_line(None, '--host')      : cp.cdb_host.setValue(host)
        if is_in_command_line(None, '--port')      : cp.cdb_port.setValue(port)
        if is_in_command_line('-e', '--experiment'): cp.exp_name.setValue(exp)
        if is_in_command_line('-d', '--detector')  : cp.data_source.setValue(det)
        #if is_in_command_line('-l', '--loglevel')  : cp.log_level.setValue(loglevel)
        #if is_in_command_line('-o', '--repodir')   : cp.log_prefix.setValue(repodir)

        cp.save_cp_at_exit.setValue(savecfg)

        if loglevel == 'DEBUG':
            print(40*'_')
            print_kwargs(kwa)


    def set_tool_tips(self):
        self.setToolTip('Calibration Management GUI')


    def set_style(self):
        self.setMinimumSize(500, 400)
        self.layout().setContentsMargins(0,0,0,0)
        spl_pos = cp.main_vsplitter.value()
        w_height = self.main_win_height.value()
        self.vspl.setSizes((spl_pos, w_height-spl_pos,))


    def closeEvent(self, e):
        logger.debug('closeEvent')
        #try: self.wspe.close()
        #except: pass
        self.wtab.close()
        self.on_save()
        QWidget.closeEvent(self, e)
        cp.wlog = None


    def key_usage(self):
        return 'Keys:'\
               '\n  V - view/hide tabs'\
               '\n'


    if __name__ == "__main__":
      def keyPressEvent(self, e):
        logger.debug('keyPressEvent, key=%s' % e.key())
        if   e.key() == Qt.Key_Escape: self.close()
        elif e.key() == Qt.Key_V: self.wtab.view_hide_tabs()
        else: logger.info(self.key_usage())


    def on_save(self):

        point, size = self.mapToGlobal(QPoint(-5,-22)), self.size() # Offset (-5,-22) for frame size.
        x,y,w,h = point.x(), point.y(), size.width(), size.height()
        msg = 'save window geometry x,y,w,h: %d,%d,%d,%d' % (x,y,w,h)
        logger.info(msg)
        print(msg)

        #Save main window position and size
        self.main_win_pos_x .setValue(x)
        self.main_win_pos_y .setValue(y)
        self.main_win_width .setValue(w)
        self.main_win_height.setValue(h)

        spl_pos = self.vspl.sizes()[0]
        msg = 'Save main v-splitter position %d' % spl_pos
        logger.debug(msg)

        cp.main_vsplitter.setValue(spl_pos)

        if cp.save_cp_at_exit.value():
            cp.printParameters()
            cp.saveParametersInFile() # see ConfigParameters


def calibman(**kwargs):
    #import sys
    #sys.stdout = sys.stderr = open('/dev/null', 'w') # open('%s-stdout-stderr' % cp.log_file.value(), 'w')
    #logging.basicConfig(format='[%(levelname).1s] %(asctime)s L:%(lineno)03d %(message)s', datefmt='%Y-%m-%dT%H:%M:%S', level=logging.DEBUG)

    global app # to prevent crash on exit
    app = QApplication([]) #sys.argv
    w = CMWMain(**kwargs)
    #w.move(0,20)
    w.show()
    app.exec_()
    #del w
    #del app


if __name__ == "__main__":
    calibman(webint=True, loglev='DEBUG')

# EOF
