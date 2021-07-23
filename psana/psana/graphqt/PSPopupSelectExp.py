
"""
:py:class:`PSPopupSelectExp` - Popup GUI for (str) experiment selection from the list of experiments
====================================================================================================

Usage::

    # Import
    from psana.graphqt.PSPopupSelectExp import PSPopupSelectExp

    # Methods - see test

See:
    - :py:class:`PSPopupSelectExp`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2014? by Mikhail Dubrovin
Adopted for LCLS2 on 2021-07-21
 - latest version of CalibManager/src/GUIPopupSelectExp.py
"""

import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QApplication, QDialog, QListWidget, QPushButton, QListWidgetItem,\
                            QVBoxLayout, QHBoxLayout, QTabBar
from PyQt5.QtCore import Qt, QPoint, QMargins, QEvent
from PyQt5.QtGui import QFont, QColor, QCursor


def years(lst_exp):
    years = []
    for exp in lst_exp:
        year = exp[-2:]
        if year in years: continue
        if not year.isdigit(): continue
        years.append(year)
    return ['20%s'%y for y in sorted(years)]


def years_and_runs(lst_exp):
    years = []
    runs  = []
    for exp in lst_exp:
        if len(exp) != 8: continue
        year = exp[-2:]
        if year in years: continue
        if not year.isdigit(): continue
        years.append(year)

    for exp in lst_exp:
        if len(exp) != 9: continue
        run = exp[-2:]
        if run in runs: continue
        if not run.isdigit(): continue
        runs.append(run)

    return ['20%s'%y for y in sorted(years)], ['Run:%s'%r for r in sorted(runs)]


def lst_exp_for_year(lst_exp, year):
    str_year = year if isinstance(year,str) else '%4d'%year
    pattern = str_year[-2:] # two last digits if the year
    return [exp for exp in lst_exp if exp[-2:]==pattern]
  

class PSPopupSelectExp(QDialog):
    """
    """
    def __init__(self, parent=None, lst_exp=[], show_frame=False):

        QDialog.__init__(self, parent)

        self.name_sel = None
        self.list = QListWidget(parent)
        self.show_frame = show_frame

        self.fill_list(lst_exp)

        # Confirmation buttons
        #self.but_cancel = QPushButton('&Cancel') 
        #self.but_apply  = QPushButton('&Apply') 
        #cp.setIcons()
        #self.but_cancel.setIcon(cp.icon_button_cancel)
        #self.but_apply .setIcon(cp.icon_button_ok)
        #self.connect(self.but_cancel, QtCore.SIGNAL('clicked()'), self.onCancel)
        #self.connect(self.but_apply,  QtCore.SIGNAL('clicked()'), self.onApply)

        #self.hbox = QVBoxLayout()
        #self.hbox.addWidget(self.but_cancel)
        #self.hbox.addWidget(self.but_apply)
        ##self.hbox.addStretch(1)

        vbox = QVBoxLayout()
        vbox.addWidget(self.list)
        self.setLayout(vbox)

        self.list.itemClicked.connect(self.on_item_click)

        self.show_tool_tips()
        self.set_style()


    def fill_list(self, lst_exp):
        self.years, self.runs = years_and_runs(lst_exp)

        for year in self.years:
            item = QListWidgetItem(year, self.list)
            item.setFont(QFont('Courier', 14, QFont.Bold))
            item.setFlags(Qt.NoItemFlags)
            #item.setFlags(Qt.NoItemFlags ^ Qt.ItemIsEnabled ^ Qt.ItemIsSelectable)
            for exp in sorted(lst_exp_for_year(lst_exp, year)):
                if len(exp) != 8: continue
                item = QListWidgetItem(exp, self.list)
                item.setFont(QFont('Monospace', 11, QFont.Normal)) # Bold))

        for run in self.runs:
            item = QListWidgetItem(run, self.list)
            item.setFont(QFont('Courier', 14, QFont.Bold))
            item.setFlags(Qt.NoItemFlags)
            #item.setFlags(Qt.NoItemFlags ^ Qt.ItemIsEnabled ^ Qt.ItemIsSelectable)
            for exp in sorted(lst_exp_for_year(lst_exp, run)):
                if len(exp) != 9: continue
                item = QListWidgetItem(exp, self.list)
                item.setFont(QFont('Monospace', 11, QFont.Normal)) # Bold))


    def set_style(self):
        self.setWindowTitle('Select experiment')
        self.setFixedWidth(120)
        self.setMinimumHeight(600)
        if not self.show_frame:
          self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.layout().setContentsMargins(2,2,2,2)
        parent = self.parentWidget()
        if parent is None:
           self.move(QCursor.pos().__add__(QPoint(-110,-50)))
        logger.debug('use %s position for popup findow' % ('CURSOR' if parent is None else 'BUTTON'))


    def show_tool_tips(self):
        self.setToolTip('Select experiment')


    def on_item_click(self, item):
        self.name_sel = item.text()
        if self.name_sel in self.years: return # ignore selection of year
        if self.name_sel in self.runs : return # ignore selection of run
        self.accept()
        self.done(QDialog.Accepted)


#    def mousePressEvent(self, e):
#        QDialog.mousePressEvent(self, e)
#        logger.debug('mousePressEvent')


    def event(self, e):
        #logger.debug('event.type %s' % str(e.type()))
        if e.type() == QEvent.WindowDeactivate:
            logger.debug('intercepted mouse click outside popup window')
            self.reject()
            self.done(QDialog.Rejected)
        return QDialog.event(self, e)
    

    def closeEvent(self, event):
        logger.debug('closeEvent')
        self.reject()
        self.done(QDialog.Rejected)


    def selectedName(self):
        return self.name_sel

 
#    def onCancel(self):
#        logger.debug('onCancel')
#        self.reject()
#        self.done(QDialog.Rejected)

#    def onApply(self):
#        logger.debug('onApply')
#        self.accept()
#        self.done(QDialog.Accepted)


def select_experiment(parent, lst_exp, show_frame=False):
    w = PSPopupSelectExp(parent, lst_exp, show_frame)
    resp=w.exec_()
    logger.debug('responce from w.exec_(): %s' % str(resp))
    return w.selectedName()


def select_instrument_experiment(parent=None, dir_instr='/cds/data/psdm', show_frame=False):
    import os
    from psana.graphqt.QWPopupSelectItem import popup_select_item_from_list
    from psana.pyalgos.generic.PSUtils import list_of_instruments, list_of_experiments
    instrs = sorted(list_of_instruments(dir_instr))
    instr = popup_select_item_from_list(parent, instrs, min_height=200, dx=-110, dy=-50, show_frame=show_frame)
    if instr is None:
       logger.debug('instrument selection is cancelled')
       return None
    dir_exp = os.path.join(dir_instr, instr)
    logger.debug('direxp:%s' % dir_exp)
    lst_exp = list_of_experiments(dir_exp) # os.listdir(dir_exp))
    return instr, select_experiment(parent, lst_exp, show_frame)

#----------- TESTS ------------

if __name__ == "__main__":
  logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d: %(message)s', level=logging.DEBUG)
  import os

  def test_all(tname):
    lst_exp = sorted(os.listdir('/reg/d/psdm/SXR/'))
    print('years form the list of experiments', years(lst_exp))
    print('years and runs form the list of experiments', str(years_and_runs(lst_exp)))
    print('experiments for 2016:', lst_exp_for_year(lst_exp, '2016'))

    app = QApplication(sys.argv)

    exp_name = 'N/A'
    if tname == '1': exp_name = select_experiment(None, lst_exp)
    else: sys.exit('not inplemented test: %s' % tname)
    print('exp_name = %s' % exp_name)

    del app


if __name__ == "__main__":
    import sys; global sys
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'
    tname = sys.argv[1] if len(sys.argv) > 1 else '1'
    print(50*'_', '\nTest %s' % tname)
    test_all(tname)
    sys.exit('End of Test %s' % tname)

# EOF
