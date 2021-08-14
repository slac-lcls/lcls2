
"""Class :py:class:`DMQWMain` is a QWidget for File Manager of LCLS1
========================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/DMQWMain.py

    from psana.graphqt.DMQWMain import DMQWMain

See method:

Created on 2021-08-10 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)
import sys

from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QTextEdit
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QTextCursor
from psana.graphqt.CMConfigParameters import cp, dir_calib, expname_def
from psana.graphqt.DMQWList import DMQWList, uws
from psana.graphqt.DMQWControl import DMQWControl
from psana.graphqt.QWInfoPanel import QWInfoPanel

class DMQWMain(QWidget):

    def __init__(self, **kwa):

        parent = kwa.get('parent', None)
        kwa.setdefault('parent', None)
 
        QWidget.__init__(self, parent)

        cp.dmqwmain = self

        self.proc_kwargs(**kwa)

        self.winfo = QWInfoPanel()
        self.wlist = DMQWList(**kwa)
        self.wctrl = DMQWControl(**kwa)

        self.hspl = QSplitter(Qt.Horizontal)
        self.hspl.addWidget(self.wlist)
        self.hspl.addWidget(self.winfo)
        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.wctrl)
        self.vbox.addWidget(self.hspl)
        self.setLayout(self.vbox)

        self.set_style()
        self.set_tool_tips()

        self.append_info = self.winfo.append # shotcut


    def proc_kwargs(self, **kwa):
        #print_kwa(kwa)
        loglevel   = kwa.get('loglevel', 'DEBUG').upper()
        logdir     = kwa.get('logdir', './')
        savelog    = kwa.get('savelog', False)


    def set_tool_tips(self):
        self.setToolTip('File Manager for LCLS1')


    def set_style(self):
        self.layout().setContentsMargins(0,0,0,0)
        self.wctrl.setFixedHeight(40)


    def closeEvent(self, e):
        logger.debug('closeEvent')
        QWidget.closeEvent(self, e)
        cp.dmqwmain = None


    def fname_info(self, expname, runnum):
        srun = '%04d' % runnum if isinstance(runnum,int) else str(runnum)
        return 'info-%s-r%s.txt' % (expname, srun)


    def dump_info_exp_run(self, expname, runnum):
        s = 'dump_info_exp_run %s run %s info:' % (expname, str(runnum))
        lst = uws.run_table_data(expname)
        if lst is None: s += ' list of run info dicts is missing'
        else:
          for d in lst:
            if d['num']==runnum:
              s += '\n' + uws.json.dumps(d, indent=2)
              break
        self.append_info(s, self.fname_info(expname, runnum))


    def dump_info_exp_run_2(self, expname, runnum):
        s = 'dump_info_exp_run_2 %s run %s info:' % (expname, str(runnum))
        lst = uws.json_runs(expname)#, location)
        if lst is None: s += ' list of run info dicts is missing'
        else:
          for d in lst:
            if d['run_num']==runnum:
              s += '\n' + uws.json.dumps(d, indent=2)
              break

        tags = uws.list_exp_tags(expname)
        s += '\n exp: %s tags %s' % (expname, tags)
        self.append_info(s, self.fname_info(expname, runnum))


    def dump_all_run_parameters(self, expname, runnum):
        s = 'all parameters for %s run %d\n' % (expname, runnum)
        jo = uws.run_parameters(expname, runnum)
        s += uws.json.dumps(jo, indent=2)
        self.append_info(s, self.fname_info(expname, runnum))


def data_manager(**kwa):
    loglevel = kwa.get('loglevel', 'DEBUG').upper()
    intlevel = logging._nameToLevel[loglevel]
    logging.basicConfig(format='[%(levelname).1s] %(name)s L%(lineno)04d: %(message)s', level=intlevel)

    a = QApplication(sys.argv)
    w = DMQWMain(**kwa)
    w.setGeometry(10, 100, 1000, 800)
    w.move(50,20)
    w.show()
    w.setWindowTitle('Data Manager')
    a.exec_()
    del w
    del a


if __name__ == "__main__":
    import os
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'
    kwa = {\
      'loglevel':'DEBUG',\
      'expname':expname_def(),\
    }

    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    if tname == '0': data_manager(**kwa)
    else: logger.debug('Not-implemented test "%s"' % tname)

# EOF
