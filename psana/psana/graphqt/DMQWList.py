
"""Class :py:class:`QWList` is a QListView->QWidget for list model
===================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/QWList.py

    from psana.graphqt.DMQWList import DMQWList
    w = QWList()

Created on 2021-08-09 by Mikhail Dubrovin
"""
import os

from psana.graphqt.QWList import *
from psana.graphqt.CMConfigParameters import cp, dir_exp
import psana.pyalgos.generic.PSUtils as psu
import psana.graphqt.UtilsWebServ as uws

logger = logging.getLogger(__name__)

EXPNAME_TEST='cxi78513'


class DMQWList(QWList):
    """Widget for the list of QWList-> QListView -> QWidget
    """
    def __init__(self, **kwa):
        QWList.__init__(self, **{})
        cp.dmqwlist = self
        cp.last_selected_run = None


    def list_runs_fs(self, expname=EXPNAME_TEST, dirinstr=psu.INSTRUMENT_DIR):
        dir_xtc = psu.dir_xtc(expname, dirinstr)
        return  psu.list_of_int_from_list_of_str(\
                  psu.list_of_runs_in_xtc_dir(dir_xtc, ext='.xtc'))


#    def dict_runinfo_db(self, expname=EXPNAME_TEST, location='SLAC'):
#        return uws.run_info(expname, location)

#    def runnums_with_tag(self, expname, tag='DARK'):
#        return uws.runnums_with_tag(expname, tag)


    def fill_list_model(self, **kwa):

        expdef = EXPNAME_TEST if cp.exp_name.is_default() else\
                 cp.exp_name.value()

        expname = self.expname  = kwa.get('experiment', expdef)
        location = kwa.get('location', 'SLAC')
        dirinstr = kwa.get('dirinstr', psu.INSTRUMENT_DIR)

        self.clear_model()
        brush_green = QBrush(Qt.green)
        runs_fs = self.list_runs_fs(expname, dirinstr)
        self.runinfo_db = uws.run_info(expname, location) # list_runs_db
        dark_runs = uws.runnums_with_tag(expname, tag='DARK')
        self.setSpacing(1)
        for r in sorted(self.runinfo_db.keys()):
            tb, te, is_closed, all_present = self.runinfo_db[r]
            in_fs = r in runs_fs
            is_dark = r in dark_runs
            s = 'run %04d %s in ARC' % (r, tb)
            s += '|FS' if in_fs else ''
            if r in dark_runs: s += ' DARK'
            item = QStandardItem(s)
            item.setAccessibleText('%d'%r)
            if in_fs:
               item.setBackground(brush_green)
            #item.setCheckable(False) 
            item.setSelectable(in_fs)
            item.setEnabled(in_fs)
            #item.setEditable(False)
            #item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            #item.setSizeHint(QSize(-1,20))
            #item.setIcon(icon.icon_table)
            self.model.appendRow(item)



    def on_click(self, index):
        item = self.model.itemFromIndex(index)
        runnum = int(item.accessibleText())
        cp.last_selected_run = runnum
        logger.info('clicked on run:%d txt:%s' % (runnum, item.text()))
        if cp.dmqwmain is None: return

        tb, te, is_closed, all_present = self.runinfo_db[runnum]
        s = '%s\nrun %04d in ARC|FS' % (80*'_', runnum)
        s += '\nbegin_time: %s\n   end_time: %s' % (tb, te)\
          + '\n is_closed: %s all_present: %s' % (is_closed, all_present)
        cp.dmqwmain.append_info(s, cp.dmqwmain.fname_info(self.expname, runnum))
        cp.dmqwmain.dump_info_exp_run(self.expname, runnum)


    def closeEvent(self, e):
        QWList.closeEvent(self, e)
        cp.dmqwlist = None
        cp.last_selected_run = None


    if __name__ == "__main__":

      def key_usage(self):
        return 'Keys:'\
               '\n  ESC - exit'\
               '\n  S - show selected items'\
               '\n'


      def keyPressEvent(self, e):
        logger.info('keyPressEvent, key=%s' % e.key())       
        if   e.key() == Qt.Key_Escape:
            self.close()

        elif e.key() == Qt.Key_S: 
            self.process_selected_items()

        else:
            logger.info(self.key_usage())


if __name__ == "__main__":
    import sys
    logging.basicConfig(format='[%(levelname).1s] L:%(lineno)03d %(name)s %(message)s', level=logging.DEBUG)

    app = QApplication(sys.argv)
    w = DMQWList()
    w.setGeometry(10, 25, 400, 600)
    w.setWindowTitle('DMQWList')
    w.move(100,50)
    w.show()
    app.exec_()
    del w
    del app

# EOF
