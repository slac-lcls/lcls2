
"""Class :py:class:`MEDControl` is a QWidget with control fields for Mask Editor
================================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/MEDControl.py

    from psana.graphqt.MEDControl import MEDControl
    w = MEDControl()

Created on 2023-09-07 by Mikhail Dubrovin
"""

import os
import sys

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout,\
                            QPushButton, QLabel, QComboBox, QLineEdit, QTextEdit
from PyQt5.QtCore import QSize, QRectF, pyqtSignal, QModelIndex, QTimer

from psana.graphqt.MEDUtils import *
from psana.graphqt.Styles import style

import psana.graphqt.GWROIUtils as roiu
from psana.graphqt.QWPopupSelectItem import popup_select_item_from_list
from psana.graphqt.QWPopupEditConfirm import popup_edit_and_confirm
from psana.graphqt.QWPopupFileName import popup_file_name
import psana.graphqt.MEDUtils as mu


class MEDControl(QWidget):
    """QWidget with control fields for Mask Editor"""

    def __init__(self, **kwa):

        #d = DIR_DATA_TEST + '/misc/'
        #kwa.setdefault('dirs', dirs_to_search())

        QWidget.__init__(self, None)

        self.def_det = 'Select'
        self.def_dbg = 'Select'

        self.geo      = kwa.get('geo', None)
        self.geofname = kwa.get('geofname', 'geometry.txt')
        self.ndafname = kwa.get('ndafname', 'ndarray.npy')
        self.dskwargs = kwa.get('dskwargs', None)
        self.detname  = kwa.get('detname', self.def_det)
        self.wmain    = kwa.get('parent', None)

        if self.wmain is not None:
            self.wisp = self.wmain.wisp
            self.wimax = self.wmain.wisp.wimax
            self.wspec = self.wmain.wisp.wspec
            self.wim = self.wmain.wisp.wimax.wim

        self.lab_geo = QLabel('Geo file:')
        self.but_geo = QPushButton(str(self.geofname))

        self.lab_nda = QLabel('N-d array:')
        self.but_nda = QPushButton(str(self.ndafname))

        self.lab_dsk = QLabel('DataSource:')
        self.but_dsk = QPushButton(str(self.dskwargs))

        self.lab_det = QLabel('Detector:')
        self.but_det = QPushButton(str(self.detname))

        self.lab_dbg = QLabel('Geo from DB:')
        self.but_dbg = QPushButton(self.def_dbg)

        self.but_set = QPushButton('Settings')
        self.but_tst = QPushButton('Test')
        self.but_mol = QPushButton('Less')

        self.list_of_buts = (
          self.but_nda,
          self.but_det,
          self.but_geo,
          self.but_dsk,
          self.but_dbg,
          self.but_set,
          self.but_mol,
          self.but_tst,
        )

        self.hbox0 = QHBoxLayout()
        self.hbox0.addWidget(self.lab_dsk)
        self.hbox0.addWidget(self.but_dsk)
        self.hbox0.addWidget(self.lab_det)
        self.hbox0.addWidget(self.but_det)
        self.hbox0.addWidget(self.lab_dbg)
        self.hbox0.addWidget(self.but_dbg)
        self.hbox0.addStretch()
        self.hbox0.addWidget(self.but_set)

        self.hbox1 = QHBoxLayout()
        self.hbox1.addWidget(self.lab_nda)
        self.hbox1.addWidget(self.but_nda)
        self.hbox1.addWidget(self.lab_geo)
        self.hbox1.addWidget(self.but_geo)
        self.hbox1.addStretch()
        self.hbox1.addWidget(self.but_mol)
        self.hbox1.addWidget(self.but_tst)

        self.vbox = QVBoxLayout()
        self.vbox.addLayout(self.hbox0)
        self.vbox.addLayout(self.hbox1)
        self.setLayout(self.vbox)

        #for but in self.list_of_buts:
        #    but.clicked.connect(self.on_but_clicked)

        self.but_dsk.clicked.connect(self.on_but_dsk)
        self.but_det.clicked.connect(self.on_but_det)
        self.but_nda.clicked.connect(self.on_but_nda)
        self.but_geo.clicked.connect(self.on_but_geo)
        self.but_dbg.clicked.connect(self.on_but_dbg)
        self.but_set.clicked.connect(self.on_but_set)
        self.but_mol.clicked.connect(self.on_but_mol)
        self.but_tst.clicked.connect(self.on_but_tst)

        self.set_tool_tips()
        self.set_style()
        self.set_visible_fields()
        self.on_but_mol()

    def set_visible_fields(self):
        self.lab_dbg.setVisible(True)
        self.but_dbg.setVisible(True)
        self.lab_geo.setVisible(True)
        self.but_geo.setVisible(True)

    def set_style(self):
        self.layout().setContentsMargins(5,5,5,5)
        self.lab_dsk.setStyleSheet(style.styleLabel)
        self.lab_det.setStyleSheet(style.styleLabel)
        self.lab_dbg.setStyleSheet(style.styleLabel)
        self.lab_nda.setStyleSheet(style.styleLabel)
        self.lab_geo.setStyleSheet(style.styleLabel)
        self.but_nda.setMaximumWidth(300)
        self.but_geo.setMaximumWidth(300)
        for but in self.list_of_buts: but.setStyleSheet(style.styleButton)
        self.but_nda.setStyleSheet(style.styleButton + 'text-align: right;')
        self.but_geo.setStyleSheet(style.styleButton + 'text-align: right;')  # style.styleButtonLeft
        #self.set_buttons_visiable()

    def set_tool_tips(self):
        self.but_nda.setToolTip('image N-d array file name')
        self.but_geo.setToolTip('Geometry file name')
        self.but_set.setToolTip('Set parameters of this app')
        self.but_dbg.setToolTip('Click and select geometry document\nNote: DB should be set.')
        self.but_tst.setToolTip('Test for development only')

    def on_but_geo(self):
        logger.debug('on_but_geo')  # sys._getframe().f_code.co_name
        path = popup_file_name(parent=self, mode='r', path=self.geofname, dirs=[], fltr='*.txt *.data\n*')
        logger.debug('Selected: %s' % str(path))
        if path is None: return
        self.but_geo.setText(path)
        #self.but_geo.setStyleSheet(style.styleButtonLeftGood)
        self.set_image()

    def on_but_nda(self):
        logger.debug('on_but_nda')  # sys._getframe().f_code.co_name
        path = popup_file_name(parent=self, mode='r', path=self.ndafname, dirs=[], fltr='*.npy *.txt *.data\n*')
        logger.debug('Selected: %s' % str(path))
        if path is None or path=='':
            self.but_nda.setText('None')
            return
        self.but_nda.setText(path)
        #self.but_nda.setStyleSheet(style.styleButtonLeftGood)
        self.set_image()

    def set_image(self):
        img, self.geo = mu.image_from_kwargs(\
               geofname=str(self.but_geo.text()),\
               ndafname=str(self.but_nda.text())
        )
        self.wim.set_pixmap_from_arr(img, set_def=True)

    def selected_none(self, sel, msg='Selection terminated'):
        resp = sel is None
        if resp:
            logger.warning('%s' % msg)
            self.but_dbc.setText(self.bud_dbc_text_def)
        return resp

    def on_but_mol(self):
        """switch button states between More and Less"""
        logger.debug('on_but_mol')
        s = 'Less' if self.but_mol.text() == 'More' else 'More'
        self.but_mol.setText(s)
        for f in (self.but_geo, self.lab_geo): f.setVisible(s=='Less')

    def on_but_dbg(self):
        logger.debug('on_but_dbg')

        exp = self.experiment_from_dskwargs()
        colname = self.but_det.text()
        if colname == self.def_det:
            logger.warning('DETECTOR IS NOT SELECTED')
            return

        dbname = 'cdb_%s' % (colname if exp is None else exp)
        self.set_geometry(dbname, colname)







    def on_but_dsk(self):
        """Select DataSource kwargs from sequence of pop-up windows."""
        logger.debug('on_but_dsk')
        self.txt_dsk_old = s = self.but_dsk.text()
        dskwa = {} if s=='Select' else mu.datasource_kwargs_from_string(s)
        logger.info('dskwargs: %s' % str(dskwa))

        dbnames = mu.db_names(fltr=None)
        insts = mu.db_instruments(dbnames)
        names1 = ('Instruments:',) + tuple(insts)
        logger.debug('Select from instruments: %s' % str(names1))
        instr = popup_select_item_from_list(self.but_dsk, names1, dx=10, dy=-10, do_sorted=False)
        if self.selected_none(instr, msg='Selection of instrument terminated'): return

        names2 = mu.db_expnames(dbnames, fltr=instr)
        names2 = ('Select experiment:',) + tuple(names2)
        logger.debug('Select from DB experiments: %s' % str(names2))
        exp = popup_select_item_from_list(self.but_dsk, names2, dx=10, dy=-10, do_sorted=False)
        if self.selected_none(exp, msg='Selection of experiment terminated'): return
        run = dskwa.get('run', 9999)
        resp, run = popup_edit_and_confirm(parent=self, msg=str(run), win_title='Run:')
        dskwa['exp'] = exp
        dskwa['run'] = run
        #if dskwa.get('files', None) == None: dskwa.pop('files')
        txt_dsk = mu.datasource_kwargs_to_string(**dskwa)
        self.but_dsk.setText(txt_dsk)
        if txt_dsk != self.txt_dsk_old:
            self.but_det.setText(self.def_det)
            self.but_dbg.setText(self.def_dbg)

    def experiment_from_dskwargs(self):
        dskwa = mu.datasource_kwargs_from_string(self.but_dsk.text())
        logger.info('dskwargs: %s' % str(dskwa))
        return dskwa.get('exp', None)

    def on_but_det(self):
        logger.debug('on_but_det')
        self.txt_det_old = self.but_dsk.text()
        exp = self.experiment_from_dskwargs()
        colname = None
        dbname = 'cdb_%s' % exp
        dbnames = mu.db_names(fltr=None)
        if dbname in dbnames: # select detector from experiment DB
            logger.info('select from detectors for exp-db: %s' % dbname)
            colls = mu.collection_names(dbname)
            for n in ('fs.chunks', 'fs.files'): colls.remove(n)
            colls = ('Select detector:',) + tuple(colls)
            logger.info(str(colls))
            colname = popup_select_item_from_list(self.but_det, colls, dx=10, dy=-10, do_sorted=False)

        else:
            logger.warning('NOT FOUND EXPERIMENT DB: %s' % dbname\
              +'\n  TO USE EXPERIMENT DB FIRST SELECT EXISTING DataSource, othervise continue with detector DB')
            #self.but_dsk.setText('Select')
            dettypes = mu.db_dettypes(dbnames)
            names1 = ('Det types:',) + tuple(dettypes)
            logger.info('Select from dettypes: %s' % str(names1))
            sel1 = popup_select_item_from_list(self.but_det, names1, dx=10, dy=-10, do_sorted=False)
            if self.selected_none(sel1, msg='Selection of dettype terminated'): return
            names2 = mu.db_detnames(dbnames, fltr=sel1)
            names2 = ('Select detector:',) + tuple(names2)
            logger.debug('dbnames to select: %s' % str(names2))
            colname = popup_select_item_from_list(self.but_det, names2, dx=10, dy=-10, do_sorted=False)
            dbname = 'cdb_%s' % colname

        if self.selected_none(colname, msg='Selection of detector terminated'): return
        self.but_det.setText(colname)
        if colname != self.txt_det_old:
            self.but_dbg.setText(self.def_dbg)

        self.set_geometry(dbname, colname)


    def set_geometry(self, dbname, colname):
        logger.info('TBD set_geometry for dbname: %s colname: %s' % (dbname, colname))
        docs = mu.find_docs(dbname, colname)
        docs_geo = [d for d in docs if d['ctype']=='geometry']
        recs = ['run:%d time_stamp:%s id:%s' % (d['run'], d['time_stamp'], d['_id']) for d in docs_geo]
        if len(recs) == 0:
            logger.warning('GEOMETRY CONSTANTS NOT FOUND for dbname: %s colname: %s' % (dbname, colname))
            self.but_dbg.setText(self.def_dbg)
            return

        recs = ('Select geometry constants:',) + tuple(recs)
        rec = popup_select_item_from_list(self.but_dbg, recs, dx=10, dy=-10, do_sorted=False)
        if self.selected_none(rec, msg='Selection of document is terminated'): return
        logger.info('selected: %s' % rec)
        run, time_stamp, _id = rec.split(' ')
        _id = _id[3:]
        doc = None
        for d in docs_geo:
            if d['_id']==_id:
                doc = d
                break
        logger.debug('doc: %s' % str(doc))
        s = 'run:%d %s' % (doc['run'], doc['time_stamp'])
        self.but_dbg.setText(s)
        geo_txt = mu.get_data_for_doc(dbname, doc)
        logger.info('selected geometry:\n%s' % geo_txt)



    def on_but_tst(self):
        logger.debug('on_but_tst')
        #dbnames = mu.db_names()
        #s = self.but_dbc.text()
        #if s == self.bud_dbc_text_def:
        #    logger.warning('DB and collection names are not selected')
        #    return
        dbname, colname = s.split()
        dbname = dbname.strip('DB:')
        colname = colname.strip('col:')
        logger.debug('dbname: %s' % str(dbname))
        logger.debug('colname: %s' % str(colname))
        #colnames = mu.collection_names(dbname)
        #logger.debug('colnames: %s' % str(colnames))

        docs = mu.find_docs(dbname, colname)
        for d in docs:
            print(d)

    def on_but_set(self):
        logger.info('on_but_set - TBD')

    def closeEvent(self, e):
        QWidget.closeEvent(self, e)

#    def on_but_dbc(self):
#        dbnames = mu.db_names(fltr=None)
#        insts = mu.db_instruments(dbnames)
#        dettypes = mu.db_dettypes(dbnames)
#        names1 = ('Instruments:',) + tuple(insts) + ('Det types:',) + tuple(dettypes)
#
#        logger.debug('Select from instruments or dettypes: %s' % str(names1))
#        sel1 = popup_select_item_from_list(self.but_dbc, names1, dx=10, dy=-10, do_sorted=False)
#        if self.selected_none(sel1): return
#        self.but_dbc.setText(sel1)
#        is_instrument = sel1 in insts
#
#        names2 = mu.db_expnames(dbnames, fltr=sel1) if is_instrument else\
#                 mu.db_detnames(dbnames, fltr=sel1)
#        names2 = ('Select DB:',) + tuple(names2)
#        logger.debug('dbnames to select: %s' % str(names2))
#        sel2 = popup_select_item_from_list(self.but_dbc, names2, dx=10, dy=-10, do_sorted=False)
#        if self.selected_none(sel2): return
#
#        dbname = 'cdb_' + sel2
#        colname = sel2
#        if is_instrument:
#            names3 = mu.collection_names(dbname)
#            for n in ('fs.chunks', 'fs.files'): names3.remove(n)
#            if len(names3) == 1:
#                colname = names3[0]
#            elif len(names3) > 1:
#                names3 = ('Select detector:',) + tuple(names3)
#                logger.debug('collections to select: %s' % str(names3))
#                sel3 = popup_select_item_from_list(self.but_dbc, names3, dx=10, dy=-10, do_sorted=False)
#                if self.selected_none(sel3, msg='Selection of collection is terminated'): return
#                colname = sel3
#            else:
#                self.selected_none(None, msg='EMPTY list of collections')
#                return
#
#        self.but_dbc.setText('DB:%s col:%s' % (dbname, colname))

#    def on_but_dbc(self):
#        dbnames = mu.db_names(fltr=None)
#        insts = mu.db_instruments(dbnames)
#        dettypes = mu.db_dettypes(dbnames)
#        names1 = ('Instruments:',) + tuple(insts) + ('Det types:',) + tuple(dettypes)
#
#        logger.debug('Select from instruments or dettypes: %s' % str(names1))
#        sel1 = popup_select_item_from_list(self.but_dbc, names1, dx=10, dy=-10, do_sorted=False)
#        if self.selected_none(sel1): return
#        self.but_dbc.setText(sel1)
#        is_instrument = sel1 in insts
#
#        names2 = mu.db_expnames(dbnames, fltr=sel1) if is_instrument else\
#                 mu.db_detnames(dbnames, fltr=sel1)
#        names2 = ('Select DB:',) + tuple(names2)
#        logger.debug('dbnames to select: %s' % str(names2))
#        sel2 = popup_select_item_from_list(self.but_dbc, names2, dx=10, dy=-10, do_sorted=False)
#        if self.selected_none(sel2): return
#
#        dbname = 'cdb_' + sel2
#        colname = sel2
#        if is_instrument:
#            names3 = mu.collection_names(dbname)
#            for n in ('fs.chunks', 'fs.files'): names3.remove(n)
#            if len(names3) == 1:
#                colname = names3[0]
#            elif len(names3) > 1:
#                names3 = ('Select detector:',) + tuple(names3)
#                logger.debug('collections to select: %s' % str(names3))
#                sel3 = popup_select_item_from_list(self.but_dbc, names3, dx=10, dy=-10, do_sorted=False)
#                if self.selected_none(sel3, msg='Selection of collection is terminated'): return
#                colname = sel3
#            else:
#                self.selected_none(None, msg='EMPTY list of collections')
#                return
#
#        self.but_dbc.setText('DB:%s col:%s' % (dbname, colname))


if __name__ == "__main__":

    #os.environ['LIBGL_ALWAYS_INDIRECT'] = '1' #export LIBGL_ALWAYS_INDIRECT=1
    logging.basicConfig(format='[%(levelname).1s] %(name)s L%(lineno)04d : %(message)s', level=logging.DEBUG)
    app = QApplication(sys.argv)
    w = MEDControl()
    w.setGeometry(100, 50, 500, 40)
    w.setWindowTitle('MED Control Panel')
    w.show()
    app.exec_()
    del w
    del app

# EOF
