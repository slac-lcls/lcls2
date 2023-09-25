
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

from psana.graphqt.MEDUtils import * #  DIR_DATA_TEST
from psana.graphqt.Styles import style

import psana.graphqt.GWROIUtils as roiu
from psana.graphqt.QWPopupSelectItem import popup_select_item_from_list
from psana.graphqt.QWPopupEditConfirm import popup_edit_and_confirm
from psana.graphqt.QWPopupFileName import popup_file_name
import psana.graphqt.MEDUtils as mu

class MEDControl(QWidget):
    """QWidget with control fields for Mask Editor"""

    def __init__(self, **kwa):
        QWidget.__init__(self, None)

        self.def_det = 'Select'
        self.def_dbg = 'Select'
        self.def_dba = 'Select'
        self.def_nda = 'Select'
        self.def_geo = 'Select'

        self.geo      = kwa.get('geo', None)
        self.geofname = kwa.get('geofname', self.def_geo) # 'geometry.txt')
        self.ndafname = kwa.get('ndafname', self.def_nda) # 'ndarray.npy')
        self.dskwargs = kwa.get('dskwargs', None)
        self.detname  = kwa.get('detname', self.def_det)
        self.wmain    = kwa.get('parent', None)

        if self.wmain is not None:
            self.wisp = self.wmain.wisp
            self.wimax = self.wmain.wisp.wimax
            self.wspec = self.wmain.wisp.wspec
            self.wim = self.wmain.wisp.wimax.wim

        self.lab_geo = QLabel('geo:')
        self.but_geo = QPushButton(str(self.geofname))

        self.lab_nda = QLabel('File array:')
        self.but_nda = QPushButton(str(self.ndafname))

        self.lab_dsk = QLabel('DataSource:')
        self.but_dsk = QPushButton(str(self.dskwargs))

        self.lab_det = QLabel('Detector:')
        self.but_det = QPushButton(str(self.detname))

        self.lab_dbg = QLabel('geo DB:')
        self.but_dbg = QPushButton(self.def_dbg)

        self.lab_dba = QLabel('array DB:')
        self.but_dba = QPushButton(self.def_dba)

        self.but_set = QPushButton('Settings')
        self.but_tst = QPushButton('Test')
        self.but_mol = QPushButton('More')

        self.hbox0 = QHBoxLayout()
        self.hbox0.addWidget(self.lab_dsk)
        self.hbox0.addWidget(self.but_dsk)
        self.hbox0.addWidget(self.lab_det)
        self.hbox0.addWidget(self.but_det)
        self.hbox0.addWidget(self.lab_dbg)
        self.hbox0.addWidget(self.but_dbg)
        self.hbox0.addWidget(self.lab_dba)
        self.hbox0.addWidget(self.but_dba)
        self.hbox0.addStretch()
        self.hbox0.addWidget(self.but_set)

        self.hbox1 = QHBoxLayout()
        self.hbox1.addWidget(self.lab_nda)
        self.hbox1.addWidget(self.but_nda)
        self.hbox1.addWidget(self.lab_geo)
        self.hbox1.addWidget(self.but_geo)
        self.hbox1.addStretch()
        self.hbox1.addWidget(self.but_tst)
        self.hbox1.addWidget(self.but_mol)

        self.vbox = QVBoxLayout()
        self.vbox.addLayout(self.hbox0)
        self.vbox.addLayout(self.hbox1)
        self.setLayout(self.vbox)

        self.list_of_buts = (
          self.but_nda,
          self.but_det,
          self.but_geo,
          self.but_dsk,
          self.but_dbg,
          self.but_dba,
          self.but_set,
          self.but_mol,
          self.but_tst,
        )

#        for but in self.list_of_buts:
#            but.clicked.connect(self.on_but_clicked)

        self.but_dsk.clicked.connect(self.on_but_dsk)
        self.but_det.clicked.connect(self.on_but_det)
        self.but_nda.clicked.connect(self.on_but_nda)
        self.but_geo.clicked.connect(self.on_but_geo)
        self.but_dbg.clicked.connect(self.on_but_dbg)
        self.but_dba.clicked.connect(self.on_but_dba)
        self.but_set.clicked.connect(self.on_but_set)
        self.but_mol.clicked.connect(self.on_but_mol)
        self.but_tst.clicked.connect(self.on_but_tst)

        #if self.geo is None and self.geofname==self.def_geo:
        #    self.set_geometry_from_kwargs() # ??? COMBINE WITH image, geo = mu.image_from_kwargs(**kwa)

        self.set_tool_tips()
        self.set_style()
        self.set_visible(is_visible=False)


    def set_style(self):
        self.layout().setContentsMargins(5,5,5,5)
        for lab in (self.lab_dsk, self.lab_det, self.lab_dbg, self.lab_dba, self.lab_nda, self.lab_geo):
            lab.setStyleSheet(style.styleLabel)
        for b in (self.but_mol, self.but_tst):
            b.setMaximumWidth(40)

        self.but_set.setMaximumWidth(60)
        self.but_nda.setMaximumWidth(300)
        self.but_geo.setMaximumWidth(300)
        self.but_dba.setMaximumWidth(190)

        for b in self.list_of_buts:
            b.setStyleSheet(style.styleButton)
        self.but_dba.setStyleSheet(style.styleButton if self.but_dba.text() == self.def_dba else style.styleButtonLeft)
        self.but_nda.setStyleSheet(style.styleButton if self.but_nda.text() == self.def_nda else style.styleButtonRight)
        self.but_geo.setStyleSheet(style.styleButton if self.but_geo.text() == self.def_geo else style.styleButtonRight)

    def set_tool_tips(self):
        self.but_nda.setToolTip('image N-d array file name')
        self.but_geo.setToolTip('Geometry file name')
        self.but_set.setToolTip('Set parameters of this app')
        self.but_dbg.setToolTip('Click and select geometry from DB\nNote: DB should be set.')
        self.but_dba.setToolTip('Click and select ndarray for image from DB\nNote: DB should be set.')
        self.but_tst.setToolTip('Test for development only')

    def on_but_geo(self):
        path0 = DIR_DATA_TEST+'/geometry' if self.geofname == self.def_nda else self.geofname
        path = popup_file_name(parent=self, mode='r', path=path0, dirs=[], fltr='*.txt *.data\n*')
        logger.debug('Selected: %s' % str(path))
        if path is None: return
        self.but_geo.setText(path)
        self.but_geo.setStyleSheet(style.styleButtonGoodRight)
        self.set_image()

    def on_but_nda(self):
        path0 = DIR_DATA_TEST+'/misc' if self.ndafname == self.def_nda else self.ndafname
        path = popup_file_name(parent=self, mode='r', path=path0, dirs=[DIR_DATA_TEST,], fltr='*.npy *.txt *.data\n*')
        logger.debug('Selected: %s' % str(path))
        if path is None or path=='':
            self.but_nda.setText('None')
            return
        self.but_nda.setText(path)
        self.but_nda.setStyleSheet(style.styleButtonGoodRight)
        self.set_image()

    def set_image(self, nda=None, geo_txt=None):
        logger.info('set_image')
        if nda is not None or geo_txt is not None: logger.warning('TBD set_image for nda, geo_txt')

        img, self.geo = mu.image_from_kwargs(\
               geofname=self.but_geo.text(),\
               ndafname=self.but_nda.text(),\
               nda=nda, geo_txt=geo_txt)
        self.wim.set_pixmap_from_arr(img, set_def=True)

    def set_visible(self, is_visible=True):
        for f in (self.lab_dba, self.but_dba, self.but_tst, self.lab_geo, self.but_geo, self.but_set):
            f.setVisible(is_visible)

    def on_but_mol(self):
        """switch button states between More and Less"""
        logger.debug('on_but_mol')
        next_visible = self.but_mol.text() == 'More'
        self.but_mol.setText('Less' if next_visible else 'More')
        self.set_visible(next_visible)

    def dbname_colname(self):
        """returns dbname, colname from button fields"""
        exp = self.experiment_from_dskwargs()
        colname = self.but_det.text()
        if colname == self.def_det:
            logger.warning('DETECTOR IS NOT SELECTED')
            return exp, None
        dbname = 'cdb_%s' % (colname if exp is None else exp)
        return dbname, colname

    def on_but_dbg(self):
        logger.info('on_but_dbg')
        dbname, colname = self.dbname_colname()
        self.set_geometry(dbname, colname)

    def on_but_dba(self):
        """sets ndarray for image from DB"""
        logger.info('on_but_dba')
        dbname, colname = self.dbname_colname()
        self.set_ndarray(dbname, colname)

    def on_but_dsk(self):
        """Select DataSource kwargs from sequence of pop-up windows."""
        self.txt_dsk_old = s = self.but_dsk.text()
        dskwa = {} if s=='Select' else mu.datasource_kwargs_from_string(s)
        logger.info('dskwargs: %s' % str(dskwa))

        dbnames = mu.db_names(fltr=None)
        insts = mu.db_instruments(dbnames)
        names1 = ('Instruments:',) + tuple(insts)
        logger.debug('Select from instruments: %s' % str(names1))
        instr = popup_select_item_from_list(self.but_dsk, names1, dx=10, dy=-10, do_sorted=False)
        if me.is_none(instr, msg='Selection of instrument terminated'): return

        names2 = mu.db_expnames(dbnames, fltr=instr)
        names2 = ('Select experiment:',) + tuple(names2)
        logger.debug('Select from DB experiments: %s' % str(names2))
        exp = popup_select_item_from_list(self.but_dsk, names2, dx=10, dy=-10, do_sorted=False)
        if me.is_none(exp, msg='Selection of experiment terminated'): return
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
        return mu.experiment_from_dskwargs(self.but_dsk.text())
        #dskwa = mu.datasource_kwargs_from_string(self.but_dsk.text())
        #logger.info('dskwargs: %s' % str(dskwa))
        #exp = dskwa.get('exp', None)
        #return dskwa.get('exp', None)

    def select_detector_in_db(self, dbname):
        logger.info('select from detectors for exp-db: %s' % dbname)
        colls = mu.collection_names(dbname)
        for n in ('fs.chunks', 'fs.files'): colls.remove(n)
        detname = None
        if len(colls) == 0:
            logger.warning('NOT FOUND ANY DETECTOR IN DB: %s' % dbname)
        elif len(colls) > 1:
            colls = ('Select detector:',) + tuple(colls)
            logger.info(str(colls))
            detname = popup_select_item_from_list(self.but_det, colls, dx=10, dy=-10, do_sorted=False)
        else:
            detname = colls[0]
        return detname

    def on_but_det(self):
        self.txt_det_old = self.but_dsk.text()
        exp = self.experiment_from_dskwargs()
        detname = None
        dbname = 'cdb_%s' % exp
        dbnames = mu.db_names(fltr=None)
        if dbname in dbnames: # select detector from experiment DB
            logger.info('select from detectors for exp-db: %s' % dbname)
            detname = self.select_detector_in_db(dbname)
        else:
            logger.warning('NOT FOUND EXPERIMENT DB: %s' % dbname\
              +'\n  TO USE EXPERIMENT DB FIRST SELECT EXISTING DataSource, othervise continue with detector DB')
            #self.but_dsk.setText('Select')
            detname = None
            dettypes = mu.db_dettypes(dbnames)
            names1 = ('Det types:',) + tuple(dettypes)
            logger.info('Select from dettypes: %s' % str(names1))
            sel1 = popup_select_item_from_list(self.but_det, names1, dx=10, dy=-10, do_sorted=False)
            if me.is_none(sel1, msg='Selection of dettype terminated'): return
            names2 = mu.db_detnames(dbnames, fltr=sel1)
            if len(names2) == 0:
                logger.warning('NOT FOUND ANY DETECTOR IN DETECTOR-DB: %s' % dbname)
            elif len(names2) > 1:
                names2 = ('Select detector:',) + tuple(names2)
                logger.debug('dbnames to select: %s' % str(names2))
                detname = popup_select_item_from_list(self.but_det, names2, dx=10, dy=-10, do_sorted=False)
            else:
                detname = names2[0]
        if me.is_none(detname, msg='Selection of detector terminated'): return

        dbname = 'cdb_%s' % detname
        self.but_det.setText(detname)
        if detname != self.txt_det_old:
            self.but_dbg.setText(self.def_dbg)
            self.but_dba.setText(self.def_dba)

        self.set_geometry(dbname, detname)



    def set_geometry_from_kwargs(self):
        s = self.dskwargs
        dskwa = {} if s=='Select' else mu.datasource_kwargs_from_string(s)
        dbname, colname = self.dbname_colname()
        if me.is_none(dbname, msg='set_geometry_from_kwargs dbname is None'): return
        run = dskwa.get('run', 9999)
        query = {'ctype':'geometry', 'run':{'$lte':run}}
        logger.info('\n==== set_geometry_from_kwargs dbname: %s colname: %s\n  query: %s' % (dbname, colname, query))
        doc = mu.find_doc(dbname, colname, query=query)
        geo_txt = mu.get_data_for_doc(dbname, doc)
        logger.info('set geometry for %s' % s)
        logger.debug('geometry constants from DB:\n%s' % geo_txt)
        self.set_but_dbg_text_for_doc(doc)

        self.set_image(geo_txt=geo_txt)


    def set_geometry(self, dbname, colname):
        logger.info('TBD set_geometry for dbname: %s colname: %s' % (dbname, colname))
        doc = None
        docs = mu.find_docs(dbname, colname)
        if me.is_none(docs, msg='docs is None for dbname: %s colname: %s' % (dbname, colname)): return
        docs_geo = [d for d in docs if d.get('ctype',None)=='geometry']
        recs = ['run:%d time_stamp:%s id:%s' % (d['run'], d['time_stamp'], d['_id']) for d in docs_geo]

        if len(recs) == 0:
            logger.warning('GEOMETRY CONSTANTS NOT FOUND for dbname: %s colname: %s' % (dbname, colname))
            self.but_dbg.setText(self.def_dbg)
            return

        elif len(recs) > 1:
            recs = ('Select geometry constants:',) + tuple(recs)
            rec = popup_select_item_from_list(self.but_dbg, recs, dx=10, dy=-10, do_sorted=False)
            if me.is_none(rec, msg='Selection of document terminated'): return
            logger.info('selected: %s' % rec)
            run, time_stamp, _id = rec.split(' ')
            _id = _id[3:]
            doc = None
            for d in docs_geo:
                if d['_id']==_id:
                    doc = d
                    break
        else:
            logger.info('ONLY ONE GEOMETRY CONSTANTS for dbname: %s colname: %s' % (dbname, colname))
            doc = docs_geo[0]

        self.set_but_dbg_text_for_doc(doc)

        geo_txt = mu.get_data_for_doc(dbname, doc)
        #logger.info('set geometry for %s' % str(doc))
        logger.debug('geometry constants from DB:\n%s' % geo_txt)

        self.set_image(geo_txt=geo_txt)

    def set_but_dbg_text_for_doc(self, doc):
        logger.debug('doc: %s' % str(doc))
        s = 'run:%d %s' % (doc['run'], doc['time_stamp'])
        self.but_dbg.setText(s)

    def set_ndarray(self, dbname, colname):
        logger.info('TBD on_but_dba: set ndarray from dbname: %s colname: %s' % (dbname, colname))
        doc = None
        docs = mu.find_docs(dbname, colname)
        docs_nda = [d for d in docs if d['ctype'] in ('pedestals', 'pixel_status', 'pixel_rms')]
        recs = ['%s run:%d time_stamp:%s id:%s' % (d['ctype'], d['run'], d['time_stamp'], d['_id']) for d in docs_nda]
        if len(recs) == 0:
            logger.warning('SUITABLE FOR NDARRAY CONSTANTS NOT FOUND for dbname: %s colname: %s' % (dbname, colname))
            self.but_dba.setText(self.def_dba)
            return
        elif len(recs) > 1:
            recs = ('Select ndarray constants:',) + tuple(sorted(recs))
            rec = popup_select_item_from_list(self.but_dba, recs, dx=10, dy=-10, do_sorted=False)
            if me.is_none(rec, msg='Selection of ndarray constants terminated'): return
            logger.info('selected: %s' % rec)
            ctype, run, time_stamp, _id = rec.split(' ')
            _id = _id[3:]
            doc = None
            for d in docs_nda:
                if d['_id']==_id:
                    doc = d
                    break
        else:
            doc = docs[0]

        logger.debug('doc: %s' % str(doc))
        s = '%s run:%d %s' % (doc['ctype'], doc['run'], doc['time_stamp'])
        self.but_dba.setText(s)
        nda = mu.get_data_for_doc(dbname, doc)
        logger.info('set ndarray as image:\n%s' % mu.info_ndarr(nda))

        self.set_image(nda=nda)

    def on_but_tst(self):
        dbnames = mu.db_names()
        print('\n==== %d cdb_* dbnames: %s\n' % (len(dbnames), str(dbnames)))
        dbname, colname = self.dbname_colname()
        colls = mu.collection_names(dbname)
        if colls is None:
          print('\n==== collections are None for dbname: %s' % (dbname))
        else:
          print('\n==== %d collections for dbname: %s\n%s' % (len(colls) if colls is not None else 0, dbname, str(colls)))

        docs = mu.find_docs(dbname, colname)
        if docs is None:
          print('\n==== docs is None for dbname: %s colname: %s' % (dbname, colname))
        else:
          print('\n==== %d docs for dbname: %s colname: %s' % (len(docs) if docs is not None else 0, dbname, colname))
          for i,d in enumerate(docs):
            print('%3d:  %12s run:%04d %s' % (i, d['ctype'], d['run'], d['time_stamp']))
            if i>100:
              print('...')
              break

    def on_but_set(self):
        logger.info('on_but_set - TBD')

    def closeEvent(self, e):
        QWidget.closeEvent(self, e)

#    def on_but_clicked(self):
#        for b in self.list_of_buts:
#            if b.hasFocus(): break
#        logger.info('click on "%s"' % b.text())
#        if   b == self.but_dsk: self.on_but_dsk()
#        elif b == self.but_det: self.on_but_det()


if __name__ == "__main__":
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
