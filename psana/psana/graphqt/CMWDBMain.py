"""
Class :py:class:`CMWDBMain` is a QWidget for interactive image
===========================================================

Usage ::

    import sys
    from PyQt5.QtWidgets import QApplication
    from psana.graphqt.CMWDBMain import CMWDBMain
    app = QApplication(sys.argv)
    w = CMWDBMain(None, app)
    w.show()
    app.exec_()

See:
    - :class:`CMWDBMain`
    - :class:`CMWMainTabs`
    - :class:`CMConfigParameters`
    - `graphqt documentation <https://lcls-psana.github.io/graphqt/py-modindex.html>`_.

Created on 2017-02-01 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-26 by Mikhail Dubrovin
"""

from PyQt5.QtWidgets import QWidget, QSplitter, QTextEdit, QVBoxLayout#, QVBoxLayout
from PyQt5.QtCore import Qt, QPoint

from psana.graphqt.CMConfigParameters import cp
from psana.pyalgos.generic.Logger import logger
from psana.graphqt.CMWDBTree import CMWDBTree
from psana.graphqt.CMWDBButtons import CMWDBButtons

#from psana.graphqt.QWIcons import icon
#from psana.graphqt.Styles import style

#------------------------------

class CMWDBMain(QWidget) :

    _name = 'CMWDBMain'

    def __init__(self, parent=None) :
        QWidget.__init__(self, parent=parent)
        #self._name = self.__class__.__name__
        cp.cmwdbmain = self

        #icon.set_icons()

        self.wbuts = CMWDBButtons()
        self.wtree = CMWDBTree()
        self.wtmp  = QTextEdit('Some text')

        #self.vbox = QVBoxLayout() 
        #self.vbox.addWidget(self.wtab) 
        #self.vbox.addStretch(1)
        #self.wrig = QWidget()
        #self.wrig.setLayout(self.vbox)

        # Horizontal splitter widget
        self.hspl = QSplitter(Qt.Horizontal)
        self.hspl.addWidget(self.wtree) 
        self.hspl.addWidget(self.wtmp) 

        # Vertical splitter widget
        self.vspl = QSplitter(Qt.Vertical) # QVBoxLayout() 
        self.vspl.addWidget(self.wbuts)
        self.vspl.addWidget(self.hspl)

        # Main box layout
        self.mbox = QVBoxLayout() 
        self.mbox.addWidget(self.vspl)
        self.setLayout(self.mbox)

        self.set_style()
        self.set_tool_tips()
        self.connect_signals_to_slots()


    def connect_signals_to_slots(self) :
        pass
        # connect button which turn on/of tabs
        #self.wbuts.but_tabs.clicked.connect(cp.cmwmaintabs.view_hide_tabs)
        #self.wbuts.but_tabs.clicked.connect(self.on_but_tabs_clicked_test)

    def on_but_tabs_clicked_test(self) :
        print('XXX: CMWDBMain.on_but_tabs_clicked')

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
            logger.info('Input image file name: %s' % ifname)
            cp.fname_img.setValue(ifname)
            cp.current_tab.setValue('File')
        #else :
        #    cp.current_tab.setValue('Data')

#------------------------------

    def set_tool_tips(self) :
        pass
        #self.butStop.setToolTip('Not implemented yet...')


    def set_style(self) :
        #self.setGeometry(self.main_win_pos_x .value(),\
        #                 self.main_win_pos_y .value(),\
        #                 self.main_win_width .value(),\
        #                 self.main_win_height.value())
        #w = self.main_win_width.value()

        self.setContentsMargins(-9,-9,-9,-9)

        self.wtree.setMinimumWidth(100)
        self.wtree.setMaximumWidth(600)

        pos_hspl = cp.cdb_hsplitter.value()
        self.hspl.setSizes((pos_hspl,1000,))

        #self.hspl.moveSplitter(200, self.hspl.indexOf(self.wtmp))

        #print('saveState:', self.hspl.saveState())
        #self.hspl.restoreState(state)

        #self.wrig.setContentsMargins(-9,-9,-9,-9)
        #self.wrig.setMinimumWidth(350)
        #self.wrig.setMaximumWidth(450)
        #self.setFixedSize(800,500)
        #self.setMinimumSize(500,800)

        #self.wrig.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        #self.hspl.moveSplitter(w*0.5,0)

        #self.setStyleSheet("background-color:blue; border: 0px solid green")
        #self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        
        #self.        setStyleSheet(style.styleBkgd)
        #self.butSave.setStyleSheet(style.styleButton)
        #self.butFBrowser.setVisible(False)

        #self.but1.raise_()


    def closeEvent(self, e) :
        logger.debug('%s.closeEvent' % self._name)

        #try : self.wspe.close()
        #except : pass

        self.on_save()

        QWidget.closeEvent(self, e)

 
    def resizeEvent(self, e):
        #logger.debug('resizeEvent', self._name) 
        #logger.info('CMWDBMain.resizeEvent: %s' % str(self.size()))
        pass


    def moveEvent(self, e) :
        #logger.debug('moveEvent', self._name) 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #logger.debug('moveEvent - pos:' + str(self.position), __name__)       
        #logger.info('CMWDBMain.moveEvent - move window to x,y: ', str(self.mapToGlobal(QPoint(0,0))))
        #self.wimg.move(self.pos() + QPoint(self.width()+5, 0))
        pass


    def keyPressEvent(self, e) :
        logger.info('%s.keyPressEvent, key=%d' % (self._name, e.key()))         
        if e.key() == Qt.Key_Escape :
            self.close()

        elif e.key() == Qt.Key_U : 
            logger.info('%s: Test:u TBD' % self._name)
            #img = image_with_random_peaks((1000, 1000))
            #self.set_image_data(img)



    def on_save(self):
        #point, size = self.mapToGlobal(QPoint(-5,-22)), self.size() # Offset (-5,-22) for frame size.
        #x,y,w,h = point.x(), point.y(), size.width(), size.height()
        #msg = 'Save main window x,y,w,h : %d, %d, %d, %d' % (x,y,w,h)
        #logger.info(msg, self._name)
        #print(msg)

        spl_pos = self.hspl.sizes()[0]
        msg = 'Save h-splitter position %d' % spl_pos

        #Save main window position and size
        cp.cdb_hsplitter.setValue(spl_pos)
        #self.main_win_pos_y .setValue(y)
        #self.main_win_width .setValue(w)
        #self.main_win_height.setValue(h)

#------------------------------

if __name__ == "__main__" :
  def test_CMWDBMain() :
    import sys
    from PyQt5.QtWidgets import QApplication
    logger.setPrintBits(0o377) 
    app = QApplication(sys.argv)
    w = CMWDBMain()
    w.setMinimumSize(600, 300)
    w.show()
    app.exec_()
    del w
    del app

#------------------------------

if __name__ == "__main__" :
    test_CMWDBMain()

#------------------------------
