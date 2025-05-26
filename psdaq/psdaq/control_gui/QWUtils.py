
"""
:py:class:`QWUtils` - Popup GUIs
================================

Usage::
    # Test: python lcls2/psdaq/psdaq/control_gui/QWUtils.py 0

    # Import
    from psdaq.control_gui.QWUtils import *

    # Methods - see test

See:
    - :py:class:`QWUtils`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2/psdaq/psdaq/control_gui>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2017-02-08 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-15
Adopted for psdaq on 2019-03-27
"""

import os
import logging
logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QMenu, QDialog, QFileDialog, QMessageBox, QColorDialog, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QCursor


def select_item_from_popup_menu(lst, title=None, default=None, parent=None):
    """Shows the list as a pop-up menu and returns the selected item as a string or None"""
    w = QMenu(parent)
    _title = title if title is not None else 'Select'
    #w.setTitle(_title)
    atitle = w.addAction(_title)
    atitle.setDisabled(True)
    asep = w.addSeparator()
    for name in lst:
       action = w.addAction(name)
       if name == default :
           #w.setDefaultAction(action)
           w.setActiveAction(action)
    item = w.exec_(QCursor.pos())
    return None if item is None else str(item.text()) # str(QString)


def select_color(colini=Qt.blue, parent=None):
    """Select color using QColorDialog"""
    qcd = QColorDialog
    w = qcd(colini, parent)
    w.setOptions(qcd.ShowAlphaChannel)# | qcd.DontUseNativeDialog | qcd.NoButtons
    res = w.exec_()
    color=w.selectedColor()
    #color = QColorDialog.getColor()
    return None if color is None else color # QColor or None


def change_check_box_list_in_popup_menu(list, win_title='Set check boxes', parent=None):
    """Shows the list of check-boxes as a dialog pop-up menu and returns the (un)changed list"""
    if list is None : return 0

    from psdaq.control_gui.QWPopupCheckList import QWPopupCheckList

    popupMenu = QWPopupCheckList(parent, list, win_title)
    #popupMenu.move(QPoint(50,50))
    popupMenu.move(QCursor.pos())
    response = popupMenu.exec_()

    if   response == QDialog.Accepted :
        #logger.debug('New checkbox list is accepted')         
        return 1
    elif response == QDialog.Rejected :
        #logger.debug('Will use old checkbox list')
        return 0
    else                                    :
        #logger.error('Unknown response...')
        return 2

#------------------------------

def change_check_box_dict_in_popup_menu(dict, win_title='Set check boxes', parent=None, msg=''):
    """Shows the dict of check-boxes as a dialog pop-up menu and returns the (un)changed dict"""
    if dict is None : return 0

    from psdaq.control_gui.QWPopupCheckDict import QWPopupCheckDict

    popupMenu = QWPopupCheckDict(parent, dict, win_title, msg)
    #popupMenu.move(QPoint(50,50))
    popupMenu.move(QCursor.pos())
    response = popupMenu.exec_()

    if   response == QDialog.Accepted :
        #logger.debug('New checkbox dict is accepted',)
        return 1
    elif response == QDialog.Rejected :
        #logger.debug('Will use old checkbox dict')
        return 0
    else                                    :
        #logger.error('Unknown response...')
        return 2


def select_radio_button_in_popup_menu(dict_of_pars, win_title='Select option', do_confirm=False, parent=None):
    """Popup GUI to select radio button from the list:  dict_of_pars = {'checked':'radio1', 'list':['radio0', 'radio1', 'radio2']}
    """
    from psdaq.control_gui.QWPopupRadioList import QWPopupRadioList

    popupMenu = QWPopupRadioList(parent, dict_of_pars, win_title, do_confirm)
    #popupMenu.move(QPoint(50,50))
    popupMenu.move(QCursor.pos()-QPoint(100,100))
    return popupMenu.exec_() # QDialog.Accepted or QDialog.Rejected


def info_point(p, cmt='', fmt='%sx=%1.0f y=%1.0f') :
    return fmt % (cmt, p.x(), p.y())

def info_rect_xywh(r, cmt='', fmt='%sx=%1.0f y=%1.0f w=%1.0f h=%1.0f') :
    return fmt % (cmt, r.x(), r.y(), r.width(), r.height())

def info_rect_lbrt(r, cmt='', fmt='%sL=%1.0f B=%1.0f R=%1.0f T=%1.0f') :
    return fmt % (cmt, r.left(), r.right(), r.top(), r.bottom())

def print_rect(r, cmt=' ') :
    logger.debug(info_rect_xywh(r, cmt))
    logger.debug(info_rect_xywh(r, cmt))
    #x, y, w, h = r.x(), r.y(), r.width(), r.height()
    #L, R, T, B = r.left(), r.right(), r.top(), r.bottom()
    #logger.debug('%s x=%8.2f  y=%8.2f  w=%8.2f  h=%8.2f' % (cmt, x, y, w, h))
    #logger.debug('%s L=%8.2f  B=%8.2f  R=%8.2f  T=%8.2f' % (len(cmt)*' ', L, B, R, T))


def get_save_fname_through_dialog_box(parent, path0, title, filter='*.txt'):

    path, fext = QFileDialog.getSaveFileName(parent,
                                             caption   = title,
                                             directory = path0,
                                             filter    = filter
                                             )
    if path == '' :
        #logger.debug('Saving is cancelled. get_save_fname_through_dialog_box')
        return None
    #logger.debug('Output file: ' + path + ' get_save_fname_through_dialog_box')
    return path


def get_open_fname_through_dialog_box(parent, path0, title, filter='*.txt'):

    path, fext = QFileDialog.getOpenFileName(parent, title, path0, filter=filter)

    #logger.debug('XXX: get_open_fname_through_dialog_box path =', path)
    #logger.debug('XXX: get_open_fname_through_dialog_box fext =', fext)

    dname, fname = os.path.split(path)
    if dname == '' or fname == '' :
        #logger.debug('Input directiry name or file name is empty... keep file path unchanged...'
        return None
    #logger.info('Input file: ' + path + 'get_open_fname_through_dialog_box')
    return path


def get_existing_directory_through_dialog_box(parent, path0, title, options = QFileDialog.ShowDirsOnly):

    resp = QFileDialog.getExistingDirectory(parent, title, path0, options)

    #logger.debug('XXX: get_open_fname_through_dialog_box path =', path)
    #logger.debug('XXX: get_open_fname_through_dialog_box fext =', fext)

    dname = resp #, fname = os.path.split(path)
    if dname == '' :
        # logger.debug('Input directiry name or file name is empty... keep file path unchanged...'
        return None
    logger.info('Selected directory: %s' % dname)
    return dname


def confirm_dialog_box(parent=None, text='Please confirm that you aware!', title='Please acknowledge') :
        """Pop-up MODAL box for confirmation"""

        mesbox = QMessageBox(parent, windowTitle=title,
                                           text=text,
                                           standardButtons=QMessageBox.Ok)
               #standardButtons=QtGui.QMessageBox.Save | QtGui.QMessageBox.Discard | QtGui.QMessageBox.Cancel)
        #mesbox.setDefaultButton(QtGui.QMessageBox.Ok)
        #mesbox.setMinimumSize(400, 200)
        #style = "background-color: rgb(255, 200, 220); color: rgb(0, 0, 100);" # Pinkish
        #style = "background-color: rgb(255, 255, 220); color: rgb(0, 0, 0);" # Yellowish
        #mesbox.setStyleSheet (style)

        clicked = mesbox.exec_() # DISPLAYS THE QMessageBox HERE
        mesbox.setDefaultButton(QMessageBox.Ok)

        #if   clicked == QtGui.QMessageBox.Save :
        #    logger.info('Saving is requested', __name__)
        #elif clicked == QtGui.QMessageBox.Discard :
        #    logger.info('Discard is requested', __name__)
        #else :
        #    logger.info('Cancel is requested', __name__)
        #return clicked

        #logger.info('You acknowkeged that saw the message:\n' + text, 'confirm_dialog_box')

        return True if clicked == QMessageBox.Ok else False


def confirm_or_cancel_dialog_box(parent=None, text='Please confirm or cancel', title='Confirm or cancel') :
        """Pop-up MODAL box for confirmation"""

        mesbox = QMessageBox(parent, windowTitle=title,
                             text=text,
                             standardButtons=QMessageBox.Ok | QMessageBox.Cancel)
               #standardButtons=QtGui.QMessageBox.Save | QtGui.QMessageBox.Discard | QtGui.QMessageBox.Cancel)
        mesbox.setDefaultButton(QMessageBox.Ok)
        mesbox.move(QCursor.pos().__add__(QPoint(-50,20)))

        #mesbox.setMinimumSize(400, 200)
        #style = "background-color: rgb(255, 200, 220); color: rgb(0, 0, 100);" # Pinkish
        #style = "background-color: rgb(255, 255, 220); color: rgb(0, 0, 0);" # Yellowish
        #mesbox.setStyleSheet (style)

        clicked = mesbox.exec_() # DISPLAYS THE QMessageBox HERE

        if   clicked == QMessageBox.Ok     : return True
        elif clicked == QMessageBox.Cancel : return False
        else                               : return False


def help_dialog_box(parent=None, text='Help message goes here', title='Help') :
        """Pop-up NON-MODAL box for help etc."""
        mesbox = QMessageBox(parent, windowTitle=title,
                                      text=text,
                                      standardButtons=QMessageBox.Ok)
                                      #standardButtons=QMessageBox.Close)
        #messbox.setStyleSheet(cp.styleBkgd)
        mesbox.setDefaultButton(QMessageBox.Ok)
        #mesbox.setWindowModality(Qt.NonModal)
        mesbox.setModal(False)
        mesbox.update()
        clicked = mesbox.exec_() # For MODAL dialog
        #clicked = mesbox.show()  # For NON-MODAL dialog
        #logger.info('Help window is open' + text, 'help_dialog_box')
        return mesbox


def widget_from_layout(l) :
    w = QWigget()
    w.setLayout(l)
    return w


def layout_from_widget(w, layout=QVBoxLayout) :
    l = layout()
    l.addWidget(w)
    return l


if __name__ == "__main__" :

  def test(tname) :

    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)

    if tname == '0':
        instrs = ['SXR', 'AMO', 'XPP', 'CXI', 'MEC']
        resp = select_item_from_popup_menu(instrs, title='Select INS', default='AMO')
        logger.debug('Selected: %s' % resp)

    elif tname == '1':
        list_of_cbox = [['VAR1', True], ['VAR2', False], ['VAR3', False], ['VAR4', False], ['VAR5', False]]
        resp = change_check_box_list_in_popup_menu(list_of_cbox, win_title='Select vars(s)')
        for (var,stat) in list_of_cbox : logger.debug('%s: %s' % (var, stat))
        logger.debug('resp: %s' % resp)

    elif tname == '2':
        dict_of_pars = {'checked':'radio1', 'list':['radio0', 'radio1', 'radio2']}
        resp = select_radio_button_in_popup_menu(dict_of_pars, win_title='Select vars(s)', do_confirm=True)
        for (k,v) in dict_of_pars.items() : logger.debug('%s: %s' % (k, v))
        logger.debug('resp: %s' % resp)

    elif tname == '3':
        parent=None; path0='./'; title='get_save_fname_through_dialog_box'
        resp = get_save_fname_through_dialog_box(parent, path0, title, filter='*.txt')
        logger.debug('resp: %s' % resp)

    elif tname == '4':
        parent=None; path0='./'; title='get_open_fname_through_dialog_box'
        resp = get_open_fname_through_dialog_box(parent, path0, title, filter='*.txt')
        logger.debug('resp: %s' % resp)

    elif tname == '5':
        resp = confirm_dialog_box(parent=None, text='Confirm that you aware!', title='Acknowledge')
        logger.debug('resp: %s' % resp)

    elif tname == '6':
        resp = confirm_or_cancel_dialog_box(parent=None, text='Confirm or cancel', title='Confirm or cancel')
        logger.debug('resp: %s' % resp)

    elif tname == '7':
        from time import sleep
        resp = help_dialog_box(parent=None, text='Help message goes here', title='Help')
        logger.debug('resp: %s' % resp)
        sleep(3)
        del resp

    elif tname == '8':
        resp = select_color(colini=Qt.blue, parent=None)

    elif tname == '9':
        dict_of_cbox = {'VAR1':True, 'VAR2':False, 'VAR3':False, 'VAR4':False, 'VAR5':False}
        resp = change_check_box_dict_in_popup_menu(dict_of_cbox, win_title='Select vars(s)')
        for (var,stat) in dict_of_cbox.items() : logger.debug('%s: %s' % (var, stat))
        logger.debug('resp: %s' % resp)

    elif tname == '10':
        resp = get_existing_directory_through_dialog_box(None, '.', 'Select directory', options = QFileDialog.ShowDirsOnly)
        logger.debug('resp: %s' % resp)

    else :
        logger.debug('Sorry, not-implemented test "%s"' % tname)

    del app


if __name__ == "__main__" :
    import sys; global sys

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    logger.debug('%s\nTest %s' % (50*'_', tname))
    test(tname)
    sys.exit('End of test %s' % tname)

# EOF
