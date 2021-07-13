
import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QWidget, QLabel, QLineEdit, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import Qt, QMargins, QRegExp, pyqtSignal
from PyQt5.QtGui import QIntValidator, QRegExpValidator

from psana.graphqt.Styles import style


class QWRange(QWidget):
    """Range setting GUI
    """
    field_is_changed = pyqtSignal('QString')

    def __init__(self, parent=None, str_from=None, str_to=None, txt_from='valid from', txt_to='to'):

        QWidget.__init__(self, None)
        self.parent = parent

        if txt_from == '':
            self.setGeometry(10, 25, 140, 40)
            self.use_lab_from = False
        else:
            self.setGeometry(10, 25, 200, 40)
            self.use_lab_from = True

        self.set_params(str_from, str_to)

        self.txt_from = txt_from
        if self.use_lab_from: self.lab_from = QLabel(txt_from)
        self.lab_to         = QLabel(txt_to)
        self.edi_from       = QLineEdit  ( self.str_from )
        self.edi_to         = QLineEdit  ( self.str_to )

        self.set_edi_validators()

        self.hboxC = QHBoxLayout()
        self.hboxC.addStretch(1)     
        if self.use_lab_from: self.hboxC.addWidget( self.lab_from )
        self.hboxC.addWidget( self.edi_from )
        self.hboxC.addWidget( self.lab_to )
        self.hboxC.addWidget( self.edi_to )
        self.hboxC.addStretch(1)     

        self.vboxW = QVBoxLayout() 
        self.vboxW.addStretch(1)
        self.vboxW.addLayout( self.hboxC ) 
        self.vboxW.addStretch(1)
        
        self.setLayout(self.vboxW)

        self.edi_from.editingFinished.connect(self.on_edi_from)
        self.edi_to  .editingFinished.connect(self.on_edi_to)
  
        self.set_tool_tips()
        self.set_style()

        # cp.guirange = self # DO NOT REGISTER THIS OBJECT! There may be many instances in the list of runs...

    def set_edi_validators(self):
        self.edi_from.setValidator(QIntValidator(0,9999,self))
        self.edi_to  .setValidator(QRegExpValidator(QRegExp("[1-9]|[1-9][0-9]|[1-9][0-9][0-9]|[1-9][0-9][0-9][0-9]|end$"),self))
        #self.edi_to  .setValidator(QRegExpValidator(QRegExp("[0-9]\\d{0,3}|end$"),self))


    def set_tool_tips(self):
        self.edi_from.setToolTip('Enter run number in range [0,9999]')
        self.edi_to  .setToolTip('Enter run number in range [1,9999] or "end"')


    def set_style(self):
        self.setStyleSheet(style.styleBkgd)

        if self.use_lab_from:
            self.setMinimumSize(200,32)
        else:
            self.setMinimumSize(100,32)

        #self.setFixedHeight(40)
        self.layout().setContentsMargins(0,0,0,0)

        self.edi_from.setFixedWidth(40)
        self.edi_to  .setFixedWidth(40)

        self.edi_from.setAlignment(Qt.AlignRight)
        self.edi_to  .setAlignment(Qt.AlignRight)

        if self.use_lab_from: self.lab_from  .setStyleSheet(style.styleLabel)
        self.lab_to.setStyleSheet(style.styleLabel)
 
        self.set_style_buttons()


    def status_buttons_is_good(self):
        if self.str_to == 'end': return True

        if int(self.str_from) > int(self.str_to):
            #msg  = 'Begin number %s exceeds the end number %s' % (self.str_from, self.str_to)
            #msg += '\nRANGE SEQUENCE SHOULD BE FIXED !!!!!!!!'
            #logger.warning(msg)            
            return False

        return True


    def set_style_buttons(self):
        if self.status_buttons_is_good():
            self.edi_from.setStyleSheet(style.styleEdit)
            self.edi_to  .setStyleSheet(style.styleEdit)
        else:
            self.edi_from.setStyleSheet(style.styleEditBad)
            self.edi_to  .setStyleSheet(style.styleEditBad)


    #def resizeEvent(self, e):
         #logger.debug('resizeEvent') 
         #pass


    #def moveEvent(self, e):
        #logger.debug('moveEvent') 
        #self.position = self.mapToGlobal(self.pos())
        #self.position = self.pos()
        #logger.debug('moveEvent - pos:' + str(self.position))       
        #pass


    #def closeEvent(self, event):
        #logger.debug('closeEvent')
        # cp.guirange = None 


#    def run( self ):
#        self.emit_field_is_changed_signal()


    def emit_field_is_changed_signal(self,msg):
        self.field_is_changed.emit(msg)

  
    def on_edi_from(self):
        #logger.debug('on_edi_from')
        txt = str( self.edi_from.text() )
        if txt == self.str_from: return # if text has not changed
        self.str_from = txt
        #msg = 'Set the range from "%s"' % self.str_from
        #logger.info(msg)
        self.set_style_buttons()
        self.emit_field_is_changed_signal('from:%s'%self.str_from)


    def on_edi_to(self):
        #logger.debug('on_edi_to')
        txt = str( self.edi_to.text() )
        if txt == self.str_to: return # if text has not changed
        self.str_to = txt
        #msg = 'Set the range up to "%s"' % self.str_to
        #logger.info(msg)
        self.set_style_buttons()
        self.emit_field_is_changed_signal('to:%s'%self.str_to)


    def set_fields_enable(self, is_enabled=True):
        """Interface method enabling/disabling the edit fields"""
        if is_enabled:
            self.set_style_buttons()
            #self.edi_from.setStyleSheet(style.styleEdit)
            #self.edi_to  .setStyleSheet(style.styleEdit)
        else:
            self.edi_from.setStyleSheet(style.styleEditInfo)
            self.edi_to  .setStyleSheet(style.styleEditInfo)

        self.edi_from.setEnabled(is_enabled) 
        self.edi_to  .setEnabled(is_enabled) 

        self.edi_from .setReadOnly(not is_enabled)
        self.edi_to   .setReadOnly(not is_enabled)


    def set_params(self, str_from=None, str_to=None):
        self.str_from = str_from if str_from is not None else '0'
        self.str_to   = str_to   if str_to is not None else 'end'


    def reset_fields(self, str_from=None, str_to=None):
        """Interface method resetting the range fields to default"""
        self.set_params(str_from, str_to)
        self.set_fields()


    def set_fields(self):
        self.edi_from.setText(self.str_from)
        self.edi_to  .setText(self.str_to)
        self.set_style_buttons()


    def range(self):
        """Interface method returning range string, for example '123-end' """
        if self.status_buttons_is_good():
            return '%d-%s' % ( int(self.str_from),
                                   self.str_to.lstrip('0') )
        else:
            return '%d-%d' % ( int(self.str_from), int(self.str_from) )

        #return self.str_from + '-' + self.str_to


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    w  = QWRange(None,'0','end','')
    w.setWindowTitle('Range setting GUI')
    w.move(10,25)
    w.show()
    app.exec_()

# EOF
