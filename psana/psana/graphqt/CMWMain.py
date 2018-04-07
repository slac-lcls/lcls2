"""
Class :py:class:`CMWMain` is a QWidget for interactive image
===========================================================

Usage ::

    import sys
    from PyQt5.QtWidgets import QApplication
    from psana.graphqt.CMWMain import CMWMain
    app = QApplication(sys.argv)
    w = CMWMain(None, app)
    w.show()
    app.exec_()

See:
    - :class:`CMWMain`
    - :class:`CMWMainTabs`
    - :class:`CMConfigParameters`
    - `graphqt documentation <https://lcls-psana.github.io/graphqt/py-modindex.html>`_.

Created on 2017-02-01 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-26 by Mikhail Dubrovin
"""
#import os
#import math

from math import floor
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QTextEdit
from PyQt5.QtGui import QPen, QBrush
from PyQt5.QtCore import Qt, QPoint

from psana.graphqt.CMConfigParameters import cp
from psana.pyalgos.generic.Logger import logger as log
from psana.graphqt.QWLogger import QWLogger

from psana.graphqt.CMWMainTabs import CMWMainTabs

#from psana.graphqt.QWUtils import selectFromListInPopupMenu

from psana.graphqt.Frame  import Frame
from psana.graphqt.QWIcons import icon
from psana.graphqt.Styles import style

#------------------------------

#class CMWMain(Frame) :
class CMWMain(QWidget) :

    _name = 'CMWMain'

    def __init__(self, parser=None) : # **dict_opts) :
        #Frame.__init__(self, parent=None, mlw=1)
        QWidget.__init__(self, parent=None)
        #self._name = self.__class__.__name__

        cp.cmwmain = self

        self.proc_parser(parser)
            
        self.main_win_width  = cp.main_win_width 
        self.main_win_height = cp.main_win_height
        self.main_win_pos_x  = cp.main_win_pos_x 
        self.main_win_pos_y  = cp.main_win_pos_y  

        #icon.set_icons()

        self.wtab = CMWMainTabs()
        self.wlog = QWLogger(log, cp, show_buttons=False)
        self.wtmp =QTextEdit('Some text')

        #self.vbox = QVBoxLayout() 
        #self.vbox.addWidget(self.wtab) 
        #self.vbox.addStretch(1)

        #self.wrig = QWidget()
        #self.wrig.setLayout(self.vbox)

        self.vspl = QSplitter(Qt.Vertical)
        #self.vspl.addWidget(self.wrig) 
        self.vspl.addWidget(self.wtab) 
        self.vspl.addWidget(self.wlog) 

        #self.hspl = QSplitter(Qt.Horizontal)
        #self.hspl.addWidget(self.vspl)
        #self.hspl.addWidget(self.wtmp)
        #self.hspl.addWidget(self.wrig)

        self.mbox = QHBoxLayout() 
        self.mbox.addWidget(self.vspl)
        self.setLayout(self.mbox)

        self.set_style()
        #self.set_tool_tips()

        self.connect_signals_to_slots()

        #self.spectrum_show(self.arr)
        self.move(self.pos()) # + QPoint(self.width()+5, 0))
        #self.wimg.show()


    def connect_signals_to_slots(self) :
        pass
        #self.connect(self.wbut.but_reset, QtCore.SIGNAL('clicked()'), self.on_but_reset)
        #self.connect(self.wbut.but_save,  QtCore.SIGNAL('clicked()'), self.on_but_save)

#------------------------------

    def proc_parser(self, parser=None) :
        self.parser=parser

        if parser is None :
            return
        return

#------------------------------

    def proc_parser_v0(self, parser=None) :
        self.parser=parser

        if parser is None :
            return

        (popts, pargs) = parser.parse_args()
        self.args = pargs
        self.opts = vars(popts)
        self.defs = vars(parser.get_default_values())

        nargs =len(self.args)

        exp = popts.exp # self.opts['exp']
        run = popts.run # self.opts['run']
        nev = popts.nev
        clb = popts.clb
        ifn = popts.ifn
        vrb = popts.vrb

        #cp.instr_dir .setValue() # val_def='/reg/d/psdm'
        if exp != self.defs['exp'] : cp.instr_name.setValue(exp[:3].upper())
        if exp != self.defs['exp'] : cp.exp_name  .setValue(exp)
        if run != self.defs['run'] : cp.str_runnum.setValue('%d'%run)
        if clb != self.defs['clb'] : cp.calib_dir .setValue(clb)

        self.verbos = vrb
 
        ifname = ifn          if ifn != self.defs['ifn'] else\
                 self.args[0] if nargs > 0 else\
                 None
        
        if ifname is not None :
            log.info('Input image file name: %s' % ifname)
            cp.fname_img.setValue(ifname)
            cp.current_tab.setValue('File')
        #else :
        #    cp.current_tab.setValue('Data')

#------------------------------

    def set_tool_tips(self) :
        pass
        #self.butStop.setToolTip('Not implemented yet...')


    def set_style(self) :
        #self.setGeometry(50, 50, 500, 600)
        self.setGeometry(self.main_win_pos_x .value(),\
                         self.main_win_pos_y .value(),\
                         self.main_win_width .value(),\
                         self.main_win_height.value())
        w_height = self.main_win_height.value()

        self.setMinimumSize(500, 400)

        w = self.main_win_width.value()

        self.setContentsMargins(-9,-9,-9,-9)

        spl_pos = cp.main_vsplitter.value()
        self.vspl.setSizes((spl_pos,w_height-spl_pos,))

        #self.wrig.setContentsMargins(-9,-9,-9,-9)
        #self.wrig.setMinimumWidth(350)
        #self.wrig.setMaximumWidth(450)

        #self.wrig.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        #self.hspl.moveSplitter(w*0.5,0)

        #self.setFixedSize(800,500)
        #self.setMinimumSize(500,800)

        #self.setStyleSheet("background-color:blue; border: 0px solid green")
        #self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        
        #self.        setStyleSheet(style.styleBkgd)
        #self.butSave.setStyleSheet(style.styleButton)
        #self.butExit.setStyleSheet(style.styleButton)
        #self.butELog.setStyleSheet(style.styleButton)
        #self.butFile.setStyleSheet(style.styleButton)

        #self.butELog    .setVisible(False)
        #self.butFBrowser.setVisible(False)

        #self.but1.raise_()


    def closeEvent(self, e) :
        log.debug('%s.closeEvent' % self._name)

        #try : self.wspe.close()
        #except : pass

        self.wtab.close()

        self.on_save()

        QWidget.closeEvent(self, e)

 
    def resizeEvent(self, e):
        #log.debug('resizeEvent', self._name) 
        #log.info('CMWMain.resizeEvent: %s' % str(self.size()))
        pass


    def moveEvent(self, e) :
        #log.debug('moveEvent', self._name) 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #log.debug('moveEvent - pos:' + str(self.position), __name__)       
        #log.info('CMWMain.moveEvent - move window to x,y: ', str(self.mapToGlobal(QPoint(0,0))))
        #self.wimg.move(self.pos() + QPoint(self.width()+5, 0))
        pass


    def key_usage(self) :
        return 'Keys:'\
               '\n  V - view/hide tabs'\
               '\n'


    def keyPressEvent(self, e) :
        #print('keyPressEvent, key=', e.key())       
        if   e.key() == Qt.Key_Escape :
            self.close()

        elif e.key() == Qt.Key_V : 
            self.wtab.view_hide_tabs()

        else :
            print(self.key_usage())



    def on_save(self):

        point, size = self.mapToGlobal(QPoint(-5,-22)), self.size() # Offset (-5,-22) for frame size.
        x,y,w,h = point.x(), point.y(), size.width(), size.height()
        msg = 'Save main window x,y,w,h : %d, %d, %d, %d' % (x,y,w,h)
        log.info(msg, self._name)
        #print(msg)

        #Save main window position and size
        self.main_win_pos_x .setValue(x)
        self.main_win_pos_y .setValue(y)
        self.main_win_width .setValue(w)
        self.main_win_height.setValue(h)

        spl_pos = self.vspl.sizes()[0]
        msg = 'XXX: Save main v-splitter position %d' % spl_pos
        print(msg)

        cp.main_vsplitter.setValue(spl_pos)

        cp.printParameters()
        cp.saveParametersInFile() # moved to PSConfigParameters

        if cp.save_log_at_exit.value() :
            log.saveLogInFile(cp.log_file.value())
            #print('Saved log file: %s' % cp.log_file.value())
            #log.saveLogTotalInFile(fnm.log_file_total())

#------------------------------

def calibman(parser=None) :
    import sys
    from PyQt5.QtWidgets import QApplication
    log.setPrintBits(0o377) 
    app = QApplication(sys.argv)
    w = CMWMain(parser)
    w.show()
    app.exec_()
    del w
    del app

#------------------------------

if __name__ == "__main__" :
    calibman()

#------------------------------
