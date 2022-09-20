
"""
:py:class:`QWDateTimeSec` - time(sec) <-> time-stamp converter
==============================================================

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

#import os
import sys

import logging
logger = logging.getLogger(__name__)

from time import time, strptime, strftime, mktime, localtime, gmtime

from PyQt5.QtWidgets import QGroupBox, QLabel, QPushButton, QLineEdit, QHBoxLayout, QApplication, QVBoxLayout
from PyQt5.QtGui import QRegExpValidator, QIntValidator
from PyQt5.QtCore import pyqtSignal, QRegExp

from psana.graphqt.Styles import style
from psana.graphqt.QWPopupSelectItem import popup_select_item_from_list


def str_tstamp(fmt='%Y-%m-%dT%H:%M:%S%Z', time_sec=None, zone='GMT'):
    """Returns string timestamp for specified format and time in sec or current time by default
    """
    struct = gmtime(time_sec) if zone=='GMT' else localtime(time_sec)
    return strftime(fmt, struct)


def get_time_sec(year, month=1, day=1, hour=0, minute=0, second=0, zone='PDT', isdst=-1):
    s_tstamp = '%04d %02d %02d %02d %02d %02d %s' % (year, month, day, hour, minute, second, zone)
    struct = strptime(s_tstamp, '%Y %m %d %H %M %S %Z')
    #struct.tm_isdst = isdst
    tsec = mktime(struct)
    return tsec


class QWDateTimeSec(QGroupBox):
    """Widget for date and time selection
    """
    year_now = int(str_tstamp(fmt='%Y', time_sec=None))
    years   = [str(y) for y in range(1970, year_now+10)]
    months  = sorted(['%02d'%m for m in range(1,13)])
    hours   = sorted(['%02d'%h for h in range(0,25)])
    minutes = sorted(['%02d'%m for m in range(0,60)])
    seconds = sorted(['%02d'%m for m in range(0,60)])
    zones   = sorted(['GMT', 'PST', 'PDT'])

    path_is_changed = pyqtSignal('QString')
    epoch_offset_sec=631152000 # for 1990-01-01 ralative 1970-01-01

    def __init__(self, parent=None):

        QGroupBox.__init__(self, 'Time converter', parent)
        self._name = self.__class__.__name__

        self.lab_month  = QLabel('-')
        self.lab_day    = QLabel('-')
        self.lab_hour   = QLabel(' ')
        self.lab_minute = QLabel(':')
        self.lab_second = QLabel(':')
        self.lab_tsec   = QLabel('<--> POSIX time sec:')
        self.lab_tsec2  = QLabel('LCLS2 time sec:')

        self.but_year   = QPushButton('2008')
        self.but_month  = QPushButton('01')
        self.but_day    = QPushButton('01')
        self.but_hour   = QPushButton('00')
        self.but_minute = QPushButton('00')
        self.but_second = QPushButton('00')
        self.but_zone   = QPushButton('PDT')

        self.edi_sec_posix = QLineEdit('1400000000')
        #self.edi_sec_posix.setValidator(QIntValidator(0, 2000000000, parent=self))
        self.edi_sec_posix.setValidator(QRegExpValidator(QRegExp('-?\d+'), parent=self))

        self.edi_sec_lcls2 = QLineEdit('0')
        #self.edi_sec_lcls2.setValidator(QIntValidator(-self.epoch_offset_sec, 2000000000, parent=self))
        self.edi_sec_lcls2.setValidator(QRegExpValidator(QRegExp('-?\d+'), parent=self))

        self.set_date_time_fields() # current time by df
        self.set_tsec()

        self.hbox = QHBoxLayout()
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.but_year)
        self.hbox.addWidget(self.lab_month)
        self.hbox.addWidget(self.but_month)
        self.hbox.addWidget(self.lab_day)
        self.hbox.addWidget(self.but_day)
        self.hbox.addWidget(self.lab_hour)
        self.hbox.addWidget(self.but_hour)
        self.hbox.addWidget(self.lab_minute)
        self.hbox.addWidget(self.but_minute)
        self.hbox.addWidget(self.lab_second)
        self.hbox.addWidget(self.but_second)
        self.hbox.addWidget(self.but_zone)
        self.hbox.addStretch(1)
        self.hbox2 = QHBoxLayout()
        self.hbox2.addStretch(1)
        self.hbox2.addWidget(self.lab_tsec)
        self.hbox2.addWidget(self.edi_sec_posix)
        self.hbox2.addStretch(1)
        self.hbox2.addWidget(self.lab_tsec2)
        self.hbox2.addWidget(self.edi_sec_lcls2)
        self.hbox2.addStretch(1)

        self.vbox = QVBoxLayout()
        self.vbox.addLayout(self.hbox)
        self.vbox.addLayout(self.hbox2)
        self.vbox.addStretch(1)

        self.setLayout(self.vbox)

        self.set_tool_tips()
        self.set_style()

        self.edi_sec_posix.textEdited.connect(self.on_edi)
        self.edi_sec_lcls2.textEdited.connect(self.on_edi_sec_lcls2)
        self.but_year  .clicked.connect(self.on_but)
        self.but_month .clicked.connect(self.on_but)
        self.but_day   .clicked.connect(self.on_but)
        self.but_hour  .clicked.connect(self.on_but)
        self.but_minute.clicked.connect(self.on_but)
        self.but_second.clicked.connect(self.on_but)
        self.but_zone  .clicked.connect(self.on_but)


    def set_tool_tips(self):
        self.setToolTip('Select date and time to get time in second')
        self.edi_sec_posix.setToolTip('POSIX epoch seconds\nfrom 1970-01-01')
        self.edi_sec_lcls2.setToolTip('LCLS2 epoch seconds\nfrom 1990-01-01')
        self.but_zone.setToolTip('Time zone\nGMT - Greenwich Mean Time'\
                     +'\nUTC - Universal Time Coordinated (GMT)'\
                     +'\nPST - Pacific Standard Time (-8:00)'\
                     +'\nPDT - Pacific Daylight Time (-7:00 during Summer)')


    def set_style(self):

        self.setStyleSheet(style.qgrbox_title)

        self.setMinimumSize(500,50)
        self.layout().setContentsMargins(5,5,5,5)

        w2d = 30
        self.but_year  .setFixedWidth(50)
        self.but_month .setFixedWidth(w2d)
        self.but_day   .setFixedWidth(w2d)
        self.but_hour  .setFixedWidth(w2d)
        self.but_minute.setFixedWidth(w2d)
        self.but_second.setFixedWidth(w2d)
        self.but_zone  .setFixedWidth(60)
        self.edi_sec_posix.setFixedWidth(100)
        self.edi_sec_lcls2.setFixedWidth(100)

        wlabel = 3
        #self.lab_year  .setFixedWidth(wlabel)
        self.lab_month .setFixedWidth(wlabel)
        self.lab_day   .setFixedWidth(wlabel)
        self.lab_hour  .setFixedWidth(wlabel)
        self.lab_minute.setFixedWidth(wlabel)
        self.lab_second.setFixedWidth(wlabel)
        self.lab_tsec  .setFixedWidth(140)
        self.lab_tsec2 .setFixedWidth(100)

        styleLabel = "color: rgb(100, 0, 150);"
        self.lab_month .setStyleSheet(styleLabel)
        self.lab_day   .setStyleSheet(styleLabel)
        self.lab_hour  .setStyleSheet(styleLabel)
        self.lab_minute.setStyleSheet(styleLabel)
        self.lab_second.setStyleSheet(styleLabel)
        self.lab_tsec  .setStyleSheet(styleLabel)
        self.lab_tsec2 .setStyleSheet(styleLabel)


    def on_edi(self, text):
        logger.debug('on_edi_sec_posix %s' % str(text))
        #txt = str(text.strip())
        txt = str(self.edi_sec_posix.displayText()).strip()
        tsec = int('0' if txt in ('','-','+') else txt)
        self.set_date_time_fields(tsec)
        self.edi_sec_lcls2.setText('%10d'%(tsec - self.epoch_offset_sec))
        self.msg_to_logger(tsec)


    def on_edi_sec_lcls2(self, text):
        logger.debug('on_edi_sec_lcls2 %s' % str(text))
        #txt = str(text.strip())
        txt = str(self.edi_sec_lcls2.displayText()).strip()
        tsec = int('0' if txt in ('','-','+') else txt) + self.epoch_offset_sec
        self.set_date_time_fields(tsec)
        self.edi_sec_posix.setText('%10d'%tsec)
        self.msg_to_logger(tsec)


    def set_date_time_fields(self, tsec=None):
        """Sets date and time fields for tsec - time in seconds or current time by default.
        """
        t_sec = int(time()) if tsec is None else tsec

        zone = str(self.but_zone.text())
        tstruct = gmtime(t_sec) if zone=='GMT' else localtime(t_sec)

        self.but_year  .setText('%4d'%tstruct.tm_year)
        self.but_month .setText('%02d'%tstruct.tm_mon)
        self.but_day   .setText('%02d'%tstruct.tm_mday)
        self.but_hour  .setText('%02d'%tstruct.tm_hour)
        self.but_minute.setText('%02d'%tstruct.tm_min)
        self.but_second.setText('%02d'%tstruct.tm_sec)
        self.but_zone  .setText(str(tstruct.tm_zone))


    def print_tsec_tstamp(self, tsec, zone='GMT'):
        logger.debug('t(sec): %d  is  %s' % (tsec, str_tstamp('%Y-%m-%dT%H:%M:%S %Z', tsec, zone)))


    def msg_to_logger(self, tsec, zone='GMT'):
        msg = 't(sec): %d  is  %s' % (tsec, str_tstamp('%Y-%m-%dT%H:%M:%S %Z', tsec, zone))
        logger.info(msg)


    def set_tsec(self):
        """Sets self.edi (tsec) field from date and time fields.
        """
        year   = int(self.but_year.text())
        month  = int(self.but_month.text())
        day    = int(self.but_day.text())
        hour   = int(self.but_hour.text())
        minute = int(self.but_minute.text())
        second = int(self.but_second.text())
        zone   = str(self.but_zone.text())
        tsec   = get_time_sec(year, month, day, hour, minute, second, zone)
        self.edi_sec_posix.setText('%10d'%tsec)
        self.edi_sec_lcls2.setText('%10d'%(tsec - self.epoch_offset_sec))
        #logger.debug('Cross-check: set t(sec): %10d for tstamp: %s' % (tsec, str_tstamp('%Y-%m-%dT%H:%M:%S', tsec, zone)))

        self.msg_to_logger(tsec, zone)
        return tsec


    def on_but(self):
        #logger.debug('on_but')
        but = None
        lst = None
        if self.but_year.hasFocus():
            but = self.but_year
            lst = self.years

        elif self.but_month.hasFocus():
            but = self.but_month
            lst = self.months

        elif self.but_day.hasFocus():
            but = self.but_day
            year  = int(self.but_year.text())
            month = int(self.but_month.text())
            days_in_feb = 29 if year%4 else 28
            days_in_month = [0, 31, days_in_feb, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            lst = ['%02d'%d for d in range(1,days_in_month[month]+1)]

        elif self.but_hour.hasFocus():
            but = self.but_hour
            lst = self.hours

        elif self.but_minute.hasFocus():
            but = self.but_minute
            lst = self.minutes

        elif self.but_second.hasFocus():
            but = self.but_second
            lst = self.seconds

        elif self.but_zone.hasFocus():
            but = self.but_zone
            lst = self.zones

        else: return

        v = popup_select_item_from_list(but, lst, dx=10, dy=-10)
        logger.debug('Selected: %s' % v)
        if v is None: return
        but.setText(v)

        tsec = self.set_tsec()
        self.set_date_time_fields(tsec)


if __name__ == "__main__":

  def test_gui(tname):
    w = QWDateTimeSec(None)
    w.setWindowTitle('Convertor of date and time to sec')
    w.show()
    app.exec_()


  def test_select_time(tname, fmt='%Y-%m-%d %H:%M:%S'):
    #lst = sorted(os.listdir('/reg/d/psdm/CXI/'))
    #logger.debug('lst: %s' % str(lst))

    ts = str_tstamp(fmt, time_sec=None)
    logger.debug('current time %s' % ts)
    year_now = int(str_tstamp(fmt='%Y', time_sec=None))
    years = [str(y) for y in range(2008, year_now+2)]
    logger.debug('years: %s' % years)

    year = int(popup_select_item_from_list(None, years))
    logger.debug('Selected year: %d' % year)

    months = sorted(['%02d'%m for m in range(1,13)])
    month = int(popup_select_item_from_list(None, months))
    logger.debug('Selected month: %d' % month)

    days_in_feb = 29 if year%4 else 28
    days_in_month = [0, 31, days_in_feb, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    days = ['%02d'%d for d in range(1,days_in_month[month]+1)]
    day = int(popup_select_item_from_list(None, days))
    logger.debug('Selected day: %d' % day)

    hours = sorted(['%02d'%h for h in range(0,25)])
    hour = int(popup_select_item_from_list(None, hours))
    logger.debug('Selected hour: %d' % hour)

    minutes = sorted(['%02d'%m for m in range(0,60)])
    minute = int(popup_select_item_from_list(None, minutes))
    logger.debug('Selected minute: %d' % minute)

    second=0

    s_tstamp = '%04d-%02d-%02d %02d:%02d:%02d' % (year, month, day, hour, minute, second)
    struct = strptime(s_tstamp, fmt)
    tsec   = mktime(struct)

    logger.debug('Input date/time : %s  time(sec) %d' % (s_tstamp, tsec))
    logger.debug('Reco ts from sec: %s' % str_tstamp(fmt, time_sec=tsec))


if __name__ == "__main__":
    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d: %(message)s', level=logging.DEBUG) #%(name)s
    tname = sys.argv[1] if len(sys.argv) > 1 else '1'
    logger.debug('%s\nTest %s' % (50*'_', tname))

    app = QApplication(sys.argv)

    if   tname == '0': test_select_time(tname)
    elif tname == '1': test_gui(tname)
    else: sys.exit('Test %s is not implemented' % tname)

    del app

    sys.exit('End of Test %s' % tname)

# EOF

