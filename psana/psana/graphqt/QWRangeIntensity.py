
""" Module QWRangeIntensity...
"""

from psana.graphqt.QWRange import *


class QWRangeIntensity(QWRange):
    """Range setting GUI
    """
    def __init__(self, parent=None, str_from=None, str_to=None, txt_from='valid from', txt_to='to'):
        QWRange.__init__(self, parent, str_from, str_to, txt_from, txt_to)


    def set_edi_validators(self):
        self.edi_from.setValidator(QRegExpValidator(QRegExp("[-+]?(\d*[.])?\d+|$"),self))
        self.edi_to  .setValidator(QRegExpValidator(QRegExp("[-+]?(\d*[.])?\d+|$"),self))


    def set_tool_tips(self):
        self.edi_from.setToolTip('Minimal value in range.\nAccepts float value or empty field for default.')
        self.edi_to  .setToolTip('Maximal value in range.\nAccepts float value or empty field for default.')


    def on_edi_from(self):
        QWRange.on_edi_from(self)
        self.plot_intens_min = self.str_from
        if self.status_buttons_is_good(): self.redraw()


    def on_edi_to(self):
        QWRange.on_edi_to(self)
        self.plot_intens_max = self.str_to
        if self.status_buttons_is_good(): self.redraw()


    def redraw(self):
        if self.parent is not None: self.parent.widgimage.on_draw()


    def set_params(self, str_from=None, str_to=None):
        self.str_from = str_from if str_from is not None else ''
        self.str_to   = str_to   if str_to   is not None else ''


    def set_style(self):
        QWRange.set_style(self)
        self.edi_from.setFixedWidth(60)
        self.edi_to  .setFixedWidth(60)


    def status_buttons_is_good(self):
        if self.str_from == '' and self.str_to == '': return True
        if self.str_from == '' or  self.str_to == '':
            #logger.warning('\nBOTH FIELDS MUST BE DEFINED OR EMPTY !!!!!!!!')
            return False
        if float(self.str_from) > float(self.str_to):
            #msg  = 'First value in range %s exceeds the second value %s' % (self.str_from, self.str_to)
            #msg += '\nRANGE SEQUENCE SHOULD BE FIXED !!!!!!!!'
            #logger.warning(msg)
            return False
        return True


if __name__ == "__main__":

    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    #w  = QWRangeIntensity(None, '-12', '345', 'From:', 'To:')
    w  = QWRangeIntensity(None, None, None, 'From:', 'To:')
    w.move(50,100)
    w.show()
    app.exec_()
    del app

# EOF
