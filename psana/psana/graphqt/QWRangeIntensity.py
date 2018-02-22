
#------------------------------
#  Module QWRangeIntensity...
#------------------------------

from psana.graphqt.QWRange import *

#------------------------------
class QWRangeIntensity(QWRange) :
    """Range setting GUI
    """

    def __init__(self, parent=None, str_from=None, str_to=None, txt_from='valid from', txt_to='to') :
        QWRange.__init__(self, parent, str_from, str_to, txt_from, txt_to)
        #super(QWRangeIntensity,self).__init__(parent, str_from, str_to, txt_from, txt_to)

    def setEdiValidators(self):
        #self.edi_from.setValidator(QDoubleValidator(-self.vmax, self.vmax, 3, self))
        #self.edi_to  .setValidator(QDoubleValidator(-self.vmax, self.vmax, 3, self))
        self.edi_from.setValidator(QRegExpValidator(QRegExp("[-+]?(\d*[.])?\d+|$"),self))
        self.edi_to  .setValidator(QRegExpValidator(QRegExp("[-+]?(\d*[.])?\d+|$"),self))


    def showToolTips(self):
        self.edi_from  .setToolTip('Minimal intensity in range.\nAccepts float value or empty field for default.')
        self.edi_to    .setToolTip('Maximal intensity in range.\nAccepts float value or empty field for default.')


    def onEdiFrom(self):
        QWRange.onEdiFrom(self)
        #cp.plot_intens_min.setValue(self.str_from)
        self.plot_intens_min = self.str_from
        if self.statusButtonsIsGood() : self.redraw()


    def onEdiTo(self):
        QWRange.onEdiTo(self)
        #cp.plot_intens_max.setValue(self.str_to)
        self.plot_intens_max = self.str_to
        if self.statusButtonsIsGood() : self.redraw()


    def redraw(self) :
        if self.parent is not None : self.parent.widgimage.on_draw()

        
    def setParams(self, str_from=None, str_to=None) :
        self.str_from = str_from if str_from is not None else ''
        self.str_to   = str_to   if str_to   is not None else ''

        #self.str_from = self.plot_intens_min
        #self.str_to   = self.plot_intens_max

        #pmin = cp.plot_intens_min
        #pmax = cp.plot_intens_max

        #if pmin.value() != pmin.value_def() : self.str_from = pmin.value()
        #if pmax.value() != pmax.value_def() : self.str_to   = pmax.value()


    def setStyle(self):
        QWRange.setStyle(self)
        #super(QWRangeIntensity,self).setStyle()
        self.edi_from.setFixedWidth(60)
        self.edi_to  .setFixedWidth(60)


    def statusButtonsIsGood(self):
        if self.str_from == '' and self.str_to == '' : return True
        if self.str_from == '' or  self.str_to == '' :
            #msg = '\nBOTH FIELDS MUST BE DEFINED OR EMPTY !!!!!!!!'
            #logger.warning(msg, __name__ )            
            return False
        if float(self.str_from) > float(self.str_to) :
            #msg  = 'First value in range %s exceeds the second value %s' % (self.str_from, self.str_to)
            #msg += '\nRANGE SEQUENCE SHOULD BE FIXED !!!!!!!!'
            #logger.warning(msg, __name__ )            
            return False
        return True

#-----------------------------

if __name__ == "__main__" :

    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    #ex  = QWRangeIntensity(None, '-12', '345', 'From:', 'To:')
    ex  = QWRangeIntensity(None, None, None, 'From:', 'To:')
    ex.move(50,100)
    ex.show()
    app.exec_()
    del app

#-----------------------------
