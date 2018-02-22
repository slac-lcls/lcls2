#------------------------------
#  Module PSPopupSelectExp...
#------------------------------

from PyQt5.QtWidgets import QDialog, QListWidget, QPushButton, QListWidgetItem,\
                            QVBoxLayout, QHBoxLayout, QTabBar
from PyQt5.QtCore import Qt, QPoint, QMargins, QEvent
from PyQt5.QtGui import QFont, QColor, QCursor

#------------------------------

def years(lst_exp) :
    years = []
    for exp in lst_exp :
        year = exp[-2:]
        if year in years : continue
        if not year.isdigit() : continue
        years.append(year)
    return ['20%s'%y for y in sorted(years)]

#------------------------------

def years_and_runs(lst_exp) :
    years = []
    runs  = []
    for exp in lst_exp :
        if len(exp) != 8 : continue
        year = exp[-2:]
        if year in years : continue
        if not year.isdigit() : continue
        years.append(year)

    for exp in lst_exp :
        if len(exp) != 9 : continue
        run = exp[-2:]
        if run in runs : continue
        if not run.isdigit() : continue
        runs.append(run)

    return ['20%s'%y for y in sorted(years)], ['Run:%s'%r for r in sorted(runs)]

#------------------------------

def lst_exp_for_year(lst_exp, year) :
    str_year = year if isinstance(year,str) else '%4d'%year
    pattern = str_year[-2:] # two last digits if the year
    return [exp for exp in lst_exp if exp[-2:]==pattern]

#------------------------------  

class PSPopupSelectExp(QDialog) :
    """
    """
    def __init__(self, parent=None, lst_exp=[]):

        QDialog.__init__(self, parent)

        self.name_sel = None
        self.list = QListWidget(parent)

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
        #vbox.addLayout(self.hbox)
        self.setLayout(vbox)

        self.list.itemClicked.connect(self.onItemClick)

        self.showToolTips()
        self.setStyle()


    def fill_list_v0(self, lst_exp) :
        for exp in sorted(lst_exp) :
            item = QListWidgetItem(exp, self.list)
        self.list.sortItems(Qt.AscendingOrder)


    def fill_list_v1(self, lst_exp) :
        self.years = sorted(years(lst_exp))
        for year in self.years :
            item = QListWidgetItem(year, self.list)
            item.setFont(QFont('Courier', 14, QFont.Bold))
            item.setFlags(Qt.NoItemFlags)
            #item.setFlags(Qt.NoItemFlags ^ Qt.ItemIsEnabled ^ Qt.ItemIsSelectable)
            for exp in sorted(lst_exp_for_year(lst_exp, year)) :
                item = QListWidgetItem(exp, self.list)
                item.setFont(QFont('Monospace', 11, QFont.Normal)) # Bold))

    def fill_list(self, lst_exp) :
        self.years, self.runs = years_and_runs(lst_exp)

        for year in self.years :
            item = QListWidgetItem(year, self.list)
            item.setFont(QFont('Courier', 14, QFont.Bold))
            item.setFlags(Qt.NoItemFlags)
            #item.setFlags(Qt.NoItemFlags ^ Qt.ItemIsEnabled ^ Qt.ItemIsSelectable)
            for exp in sorted(lst_exp_for_year(lst_exp, year)) :
                if len(exp) != 8 : continue
                item = QListWidgetItem(exp, self.list)
                item.setFont(QFont('Monospace', 11, QFont.Normal)) # Bold))

        for run in self.runs :
            item = QListWidgetItem(run, self.list)
            item.setFont(QFont('Courier', 14, QFont.Bold))
            item.setFlags(Qt.NoItemFlags)
            #item.setFlags(Qt.NoItemFlags ^ Qt.ItemIsEnabled ^ Qt.ItemIsSelectable)
            for exp in sorted(lst_exp_for_year(lst_exp, run)) :
                if len(exp) != 9 : continue
                item = QListWidgetItem(exp, self.list)
                item.setFont(QFont('Monospace', 11, QFont.Normal)) # Bold))


    def setStyle(self):
        self.setWindowTitle('Select experiment')
        self.setFixedWidth(120)
        self.setMinimumHeight(600)
        #self.setMaximumWidth(600)
        #self.setStyleSheet(cp.styleBkgd)
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setContentsMargins(QMargins(-9,-9,-9,-9))
        #self.setStyleSheet(cp.styleBkgd)
        #self.but_cancel.setStyleSheet(cp.styleButton)
        #self.but_apply.setStyleSheet(cp.styleButton)
        self.move(QCursor.pos().__add__(QPoint(-110,-50)))


    def showToolTips(self):
        #self.but_apply.setToolTip('Apply selection')
        #self.but_cancel.setToolTip('Cancel selection')
        self.setToolTip('Select experiment')


    def onItemClick(self, item):
        #if item.isSelected(): item.setSelected(False)
        #widg = self.list.itemWidget(item)
        #item.checkState()
        self.name_sel = item.text()
        if self.name_sel in self.years : return # ignore selection of year
        if self.name_sel in self.runs  : return # ignore selection of run
        #print(self.name_sel)
        #logger.debug('Selected experiment %s' % self.name_sel, __name__)  
        self.accept()


    def event(self, e):
        """Intercepts mouse clicks outside popup window"""
        #print('event.type', e.type())
        if e.type() == QEvent.WindowDeactivate :
            self.reject()
        return QDialog.event(self, e)
    

    def closeEvent(self, event):
        #logger.info('closeEvent', __name__)
        self.reject()


    def selectedName(self):
        return self.name_sel

 
    def onCancel(self):
        #logger.debug('onCancel', __name__)
        self.reject()


    def onApply(self):
        #logger.debug('onApply', __name__)  
        self.accept()

#------------------------------
#----------- TESTS ------------
#------------------------------

if __name__ == "__main__" :

  def select_experiment_v1(parent, lst_exp) :

    w = PSPopupSelectExp(parent, lst_exp)
    ##w.show()
    resp=w.exec_()
    if   resp == QDialog.Accepted : return w.selectedName()
    elif resp == QDialog.Rejected : return None
    else : return None

  #------------------------------  

  def test_all(tname) :
    import os
    from PyQt5.QtWidgets import QApplication

    lst_exp = sorted(os.listdir('/reg/d/psdm/SXR/'))
    #lst_exp = sorted(os.listdir('/reg/d/psdm/CXI/'))
    #print('lst_exps:', lst_exp)  
    print('years form the list of experiments', years(lst_exp))
    print('years and runs form the list of experiments', str(years_and_runs(lst_exp)))
    print('experiments for 2016:', lst_exp_for_year(lst_exp, '2016'))

    app = QApplication(sys.argv)

    exp_name = 'N/A'
    if tname == '1': exp_name = select_experiment_v1(None, lst_exp)

    print('exp_name = %s' % exp_name)

    del app

#------------------------------

if __name__ == "__main__" :
    import sys; global sys
    tname = sys.argv[1] if len(sys.argv) > 1 else '1'
    print(50*'_', '\nTest %s' % tname)
    test_all(tname)
    sys.exit('End of Test %s' % tname)

#------------------------------
