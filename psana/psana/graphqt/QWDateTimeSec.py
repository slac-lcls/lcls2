#------------------------------
"""
:py:class:`QWDateTimeSec` - time(sec) <-> time-stamp converter
============================================================================================

Usage::

    # Import
    from psana.graphqt.QWDateTimeSec import QWDateTimeSec

    # Methods - see test

See:
    - :py:class:`QWDateTimeSec`
    - :py:class:`QWPopupSelectItem`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2017-06-14 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-15
"""
#------------------------------

#import os
import sys
from time import time, strptime, strftime, mktime, localtime, struct_time

from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QLineEdit, QHBoxLayout, QVBoxLayout, QApplication
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import pyqtSignal

#from psana.graphqt.Frame import Frame
from psana.graphqt.Styles import style
from psana.graphqt.QWPopupSelectItem import popup_select_item_from_list

#------------------------------

def str_tstamp(fmt='%Y-%m-%dT%H:%M:%S', time_sec=None) :
    """Returns string timestamp for specified format and time in sec or current time by default
    """
    return strftime(fmt, localtime(time_sec))

#------------------------------

def time_sec(year, month=1, day=1, hour=0, minute=0, second=0) :
    s_tstamp = '%04d %02d %02d %02d %02d %02d' % (year, month, day, hour, minute, second)
    struct = strptime(s_tstamp, '%Y %m %d %H %M %S')
    tsec   = mktime(struct)
    return tsec
 
#------------------------------

class QWDateTimeSec(QWidget) : # Frame
    """Widget for date and time selection
    """
    year_now = int(str_tstamp(fmt='%Y', time_sec=None))
    years   = [str(y) for y in range(2008, year_now+2)]
    months  = sorted(['%02d'%m for m in range(1,13)])
    hours   = sorted(['%02d'%h for h in range(0,25)])
    minutes = sorted(['%02d'%m for m in range(0,60)])
    seconds = sorted(['%02d'%m for m in range(0,60)])

    path_is_changed = pyqtSignal('QString')

    def __init__(self, parent=None, show_frame=False, verb=False, logger=None) :

        QWidget.__init__(self, parent)
        #Frame.__init__(self, parent, mlw=1, vis=show_frame)
        self._name = self.__class__.__name__
        self.verb = verb
        self.set_logger(logger)

        #self.lab_year   = QLabel('')
        self.lab_month  = QLabel('-')
        self.lab_day    = QLabel('-')
        self.lab_hour   = QLabel(' ')
        self.lab_minute = QLabel(':')
        self.lab_second = QLabel(':')
        self.lab_tsec   = QLabel(' <--> t(sec):')

        self.but_year   = QPushButton('2008')
        self.but_month  = QPushButton('01')
        self.but_day    = QPushButton('01')
        self.but_hour   = QPushButton('00')
        self.but_minute = QPushButton('00')
        self.but_second = QPushButton('00')

        self.edi = QLineEdit('1400000000')
        self.edi.setValidator(QIntValidator(1000000000,2000000000,self))
        #self.edi.setReadOnly(True) 

        self.set_date_time_fields() # current time by df
        self.set_tsec()

        self.hbox = QHBoxLayout() 
        #self.hbox.addWidget(self.lab_year  )
        self.hbox.addWidget(self.but_year  )
        self.hbox.addWidget(self.lab_month )
        self.hbox.addWidget(self.but_month )
        self.hbox.addWidget(self.lab_day   )
        self.hbox.addWidget(self.but_day   )
        self.hbox.addWidget(self.lab_hour  )
        self.hbox.addWidget(self.but_hour  )
        self.hbox.addWidget(self.lab_minute)
        self.hbox.addWidget(self.but_minute)
        self.hbox.addWidget(self.lab_second)
        self.hbox.addWidget(self.but_second)
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.lab_tsec)
        self.hbox.addWidget(self.edi)
        self.hbox.addStretch(1)

        self.vbox = QVBoxLayout() 
        self.vbox.addLayout(self.hbox)
        self.vbox.addStretch(1)

        self.setLayout(self.vbox)

        self.set_tool_tips()
        self.set_style()

        self.but_year  .clicked.connect(self.on_but)
        self.but_month .clicked.connect(self.on_but)
        self.but_day   .clicked.connect(self.on_but)
        self.but_hour  .clicked.connect(self.on_but)
        self.but_minute.clicked.connect(self.on_but)
        self.but_second.clicked.connect(self.on_but)
        self.edi       .editingFinished.connect(self.on_edi)


    def set_tool_tips(self) :
        self.setToolTip('Select date and time to get time in second')
        self.edi.setToolTip('Edit seconds to get date and time')

    def set_style(self) :
        self.setMinimumSize(300,40)
        #self.setContentsMargins(-9,-9,-9,-9)
        #self.but_year  .setStyleSheet(style.styleButton)
        w2d = 30
        self.but_year  .setFixedWidth(50)
        self.but_month .setFixedWidth(w2d)
        self.but_day   .setFixedWidth(w2d)
        self.but_hour  .setFixedWidth(w2d)
        self.but_minute.setFixedWidth(w2d)
        self.but_second.setFixedWidth(w2d)
        self.edi       .setFixedWidth(100)

        wlabel = 3
        #self.lab_year  .setFixedWidth(wlabel)
        self.lab_month .setFixedWidth(wlabel)
        self.lab_day   .setFixedWidth(wlabel)
        self.lab_hour  .setFixedWidth(wlabel)
        self.lab_minute.setFixedWidth(wlabel)
        self.lab_second.setFixedWidth(wlabel)
        self.lab_tsec  .setFixedWidth(80)

        #self.lab_year  .setStyleSheet(style.styleLabel)
        self.lab_month .setStyleSheet(style.styleLabel)
        self.lab_day   .setStyleSheet(style.styleLabel)
        self.lab_hour  .setStyleSheet(style.styleLabel)
        self.lab_minute.setStyleSheet(style.styleLabel)
        self.lab_second.setStyleSheet(style.styleLabel)
        self.lab_tsec  .setStyleSheet(style.styleLabel)


    def on_edi(self):
        tsec = int(self.edi.displayText())
        self.set_date_time_fields(tsec)
        if self.verb : self.print_tsec_tstamp(tsec)
        self.msg_to_logger(tsec)


    def set_date_time_fields(self, tsec=None):
        """Sets date and time fields for tsec - time in seconds or current time by default.
        """
        t_sec = int(time()) if tsec is None else tsec
        tstruct = localtime(t_sec)
        #print('tstruct:', tstruct)
        #print('tstruct.tm_year:', tstruct.tm_year)
        #print('tstruct.tm_mday:', tstruct.tm_mday)

        #print('t(sec): %d' % t_sec)
        tstruct = localtime(t_sec)
        self.but_year  .setText('%4d'%tstruct.tm_year)
        self.but_month .setText('%02d'%tstruct.tm_mon)
        self.but_day   .setText('%02d'%tstruct.tm_mday)
        self.but_hour  .setText('%02d'%tstruct.tm_hour)
        self.but_minute.setText('%02d'%tstruct.tm_min)
        self.but_second.setText('%02d'%tstruct.tm_sec)


    def print_tsec_tstamp(self, tsec):
        print('t(sec): %d  is  %s' % (tsec, str_tstamp('%Y-%m-%dT%H:%M:%S%z', tsec)))


    def set_logger(self, logger=None):
        self.logger = logger


    def msg_to_logger(self, tsec):
        if self.logger is None : return
        msg = 't(sec): %d  is  %s' % (tsec, str_tstamp('%Y-%m-%dT%H:%M:%S%z', tsec))
        self.logger.info(msg, self._name)


    def set_tsec(self) :
        """Sets self.edi (tsec) field from date and time fields.
        """
        year   = int(self.but_year.text())
        month  = int(self.but_month.text())
        day    = int(self.but_day.text())
        hour   = int(self.but_hour.text())
        minute = int(self.but_minute.text())
        second = int(self.but_second.text())
        tsec   = time_sec(year, month, day, hour, minute, second)
        self.edi.setText('%10d'%tsec)
        #print('Cross-check: set t(sec): %10d for tstamp: %s' % (tsec, str_tstamp('%Y-%m-%dT%H:%M:%S', tsec)))

        if self.verb : self.print_tsec_tstamp(tsec)
        self.msg_to_logger(tsec)


    def on_but(self):
        #print 'on_but'
        but = None
        lst = None
        if self.but_year.hasFocus() :
            but = self.but_year
            lst = self.years

        elif self.but_month.hasFocus() :
            but = self.but_month
            lst = self.months

        elif self.but_day.hasFocus() :
            but = self.but_day
            year  = int(self.but_year.text())
            month = int(self.but_month.text())
            days_in_feb = 29 if year%4 else 28
            days_in_month = [0, 31, days_in_feb, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            lst = ['%02d'%d for d in range(1,days_in_month[month]+1)]

        elif self.but_hour.hasFocus() :
            but = self.but_hour
            lst = self.hours

        elif self.but_minute.hasFocus() :
            but = self.but_minute
            lst = self.minutes

        elif self.but_second.hasFocus() :
            but = self.but_second
            lst = self.seconds

        else : return

        v = popup_select_item_from_list(but, lst)
        #print('Selected: %s' % v)
        if v is None : return
        but.setText(v)

        self.set_tsec()

#------------------------------
#------------------------------
#----------- TESTS ------------
#------------------------------
#------------------------------
if __name__ == "__main__" :

  def test_gui(tname) :
    w = QWDateTimeSec(None, show_frame=True)
    w.setWindowTitle('Convertor of date and time to sec')
    w.show()
    app.exec_()


  def test_select_time(tname, fmt='%Y-%m-%d %H:%M:%S') :
    #lst = sorted(os.listdir('/reg/d/psdm/CXI/'))
    #print('lst:', lst)

    ts = str_tstamp(fmt, time_sec=None)
    print('current time %s' % ts)
    year_now = int(str_tstamp(fmt='%Y', time_sec=None))
    years = [str(y) for y in range(2008, year_now+2)]
    print('years: %s' % years)

    year = int(popup_select_item_from_list(None, years))
    print('Selected year: %d' % year)

    months = sorted(['%02d'%m for m in range(1,13)])
    month = int(popup_select_item_from_list(None, months))
    print('Selected month: %d' % month)

    days_in_feb = 29 if year%4 else 28
    days_in_month = [0, 31, days_in_feb, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    days = ['%02d'%d for d in range(1,days_in_month[month]+1)]
    day = int(popup_select_item_from_list(None, days))
    print('Selected day: %d' % day)

    hours = sorted(['%02d'%h for h in range(0,25)])
    hour = int(popup_select_item_from_list(None, hours))
    print('Selected hour: %d' % hour)

    minutes = sorted(['%02d'%m for m in range(0,60)])
    minute = int(popup_select_item_from_list(None, minutes))
    print('Selected minute: %d' % minute)

    second=0

    s_tstamp = '%04d-%02d-%02d %02d:%02d:%02d' % (year, month, day, hour, minute, second)
    struct = strptime(s_tstamp, fmt)
    tsec   = mktime(struct)

    print('Input date/time  : %s  time(sec) %d' % (s_tstamp, tsec))
    print('Reco ts from sec : %s' % str_tstamp(fmt, time_sec=tsec))

    #exp_name = popup_select_item_from_list(None, lst)
    #print('exp_name = %s' % exp_name)

#------------------------------

if __name__ == "__main__" :

    tname = sys.argv[1] if len(sys.argv) > 1 else '1'
    print(50*'_', '\nTest %s' % tname)

    app = QApplication(sys.argv)

    if   tname == '0': test_select_time(tname)
    elif tname == '1': test_gui(tname)
    else : sys.exit('Test %s is not implemented' % tname)

    del app

    sys.exit('End of Test %s' % tname)

#------------------------------

