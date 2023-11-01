
"""Class :py:class:`IVControlSpec` is a QWidget class for spectrum control fields
==================================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/IVControlSpec.py

    from psana.graphqt.IVControlSpec import IVControlSpec
    w = IVControlSpec()

Created on 2021-07-14 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLineEdit, QLabel, QComboBox
from PyQt5.QtCore import pyqtSignal, QRegExp
from PyQt5.QtGui import QRegExpValidator, QIntValidator
from math import floor, ceil

class IVControlSpec(QWidget):
    """QWidget - class for spectrum control fields"""

    spectrum_range_changed = pyqtSignal(dict)

    range_modes = ('value', 'fraction', 'full')
    mode_def = range_modes[1]

    nbins_def = 1000
    amin_def  = -100
    amax_def  =  100
    frmin_def = 0.001
    frmax_def = 0.999


    def __init__(self, **kwargs):

        parent = kwargs.get('parent', None)
        QWidget.__init__(self, parent)

        self.nbins = kwargs.get('nbins', self.nbins_def)
        self.amin  = kwargs.get('amin',  self.amin_def)
        self.amax  = kwargs.get('amax',  self.amax_def)
        self.frmin = kwargs.get('frmin', self.frmin_def)
        self.frmax = kwargs.get('frmax', self.frmax_def)

        self.lab_nbins = QLabel('nbins:')
        self.lab_min   = QLabel('min:')
        self.lab_max   = QLabel('max:')
        self.cmb_mode  = QComboBox(self)
        self.cmb_mode.addItems(self.range_modes)
        self.cmb_mode.setCurrentIndex(self.range_modes.index(self.mode_def))

        self.edi_nbins = QLineEdit(str(self.nbins))
        self.edi_min   = QLineEdit(str(self.frmin))
        self.edi_max   = QLineEdit(str(self.frmax))

        self.edi_nbins.setValidator(QIntValidator(1, 1000000, parent=self))
        self.edi_min.setValidator(QRegExpValidator(QRegExp("[-+]?(\d*[.])?\d+|$"), parent=self))
        self.edi_max.setValidator(QRegExpValidator(QRegExp("[-+]?(\d*[.])?\d+|$"), parent=self))
        self.box = QHBoxLayout()
        self.box.addStretch(1)
        self.box.addWidget(self.lab_min)
        self.box.addWidget(self.edi_min)
        self.box.addWidget(self.lab_max)
        self.box.addWidget(self.edi_max)
        self.box.addWidget(self.lab_nbins)
        self.box.addWidget(self.edi_nbins)
        self.box.addWidget(self.cmb_mode)
        self.setLayout(self.box)

        self.set_tool_tips()
        self.set_style()
        self.set_range_fields_visible()
        self.connect_fields()


    def connect_fields(self):
        self.cmb_mode.currentIndexChanged[int].connect(self.on_cmb_mode)
        self.edi_nbins.editingFinished.connect(self.on_editing_finished)
        self.edi_min.editingFinished.connect(self.on_editing_finished)
        self.edi_max.editingFinished.connect(self.on_editing_finished)


    def disconnect_fields(self):
        self.cmb_mode.currentIndexChanged[int].disconnect(self.on_cmb_mode)


    def connect_signal_spectrum_range_changed(self, recip):
        self.spectrum_range_changed.connect(recip)


    def disconnect_signal_spectrum_range_changed(self, recip):
        self.spectrum_range_changed.disconnect(recip)


    def on_cmb_mode(self):
        mode = str(self.cmb_mode.currentText())
        logger.debug('on_cmb_mode selected %s' % (mode))
        amin = self.frmin_def if mode == 'fraction' else self.amin_def
        amax = self.frmax_def if mode == 'fraction' else self.amax_def
        self.edi_min.setText(str(amin))
        self.edi_max.setText(str(amax))

        self.set_range_fields_visible()
        self.emit_signal_spectrum_range_changed()


    def set_amin_amax_def(self, amin=-100, amax=100):
        """it would be nice to initialize these values when new image array is loaded
        """
        self.amin_def = floor(amin)
        self.amax_def = ceil(amax)
        if self.is_mode('value'):
            self.edi_min.setText(str(self.amin_def))
            self.edi_max.setText(str(self.amax_def))


    def on_editing_finished(self):
        self.nbins_def = self.edi_nbins.text()
        if self.is_mode('value'):
          self.amin_def = self.edi_min.text()
          self.amax_def = self.edi_max.text()
        elif self.is_mode('fraction'):
          self.frmin_def = self.edi_min.text()
          self.frmax_def = self.edi_max.text()

        self.emit_signal_spectrum_range_changed()


    def emit_signal_spectrum_range_changed(self):
        d = self.spectrum_parameters()
        logger.debug('emit_signal_spectrum_range_changed %s' % str(d))
        self.spectrum_range_changed.emit(d)


    def spectrum_parameters(self):
        mode = str(self.cmb_mode.currentText())
        mode_full     = mode == 'full'
        mode_fraction = mode == 'fraction'
        mode_value    = mode == 'value'
        amin = None if mode_full else float(self.edi_min.text())
        amax = None if mode_full else float(self.edi_max.text())
        return {'mode':mode,
             'nbins':int(self.edi_nbins.text()),
             'amin':amin if mode_value else None,
             'amax':amax if mode_value else None,
             'frmin':amin if mode_fraction else None,
             'frmax':amax if mode_fraction else None
        }


    def set_tool_tips(self):
        self.edi_nbins.setToolTip('spectrum histogram\nnumber of bins')
        self.edi_min.setToolTip('spectrum range\nminimal value/fraction')
        self.edi_max.setToolTip('spectrum range\nmaximal value/fraction')


    def set_style(self):
        self.layout().setContentsMargins(5,0,5,0)
        self.edi_nbins.setFixedWidth(55)
        self.edi_min.setFixedWidth(55)
        self.edi_max.setFixedWidth(55)
        self.cmb_mode.setFixedWidth(80)


    def is_mode(self, mode='full'):
        return str(self.cmb_mode.currentText()) == mode


    def set_range_fields_visible(self, isvisible=None):
        isvisible = not self.is_mode('full') if isvisible is None else isvisible
        self.lab_min.setVisible(isvisible)
        self.lab_max.setVisible(isvisible)
        self.edi_min.setVisible(isvisible)
        self.edi_max.setVisible(isvisible)


if __name__ == "__main__":

    def test_signal_spectrum_range_changed(d):
        print(' test_signal_spectrum_range_changed:', d)

    import os
    import sys
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'
    from PyQt5.QtWidgets import QApplication
    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d: %(message)s', level=logging.DEBUG)

    app = QApplication(sys.argv)
    w = IVControlSpec()
    w.setGeometry(100, 50, 300, 30)
    w.setWindowTitle('Spectrum control')
    w.connect_signal_spectrum_range_changed(test_signal_spectrum_range_changed)
    w.show()
    app.exec_()
    del w
    del app

# EOF
