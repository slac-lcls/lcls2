#------------------------------
"""
:py:class:`QWUtils` - Popup GUIs
================================

Usage::
    # Test: python lcls2/psana/psana/graphqt/QWUtils.py 0

    # Import
    from psana.graphqt.QWUtils import *

    # Methods - see test

See:
    - :py:class:`QWUtils`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2017-02-08 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-15
"""
#------------------------------

import os
from PyQt5.QtWidgets import QMenu, QDialog, QFileDialog, QMessageBox, QColorDialog, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QCursor

#------------------------------

def select_item_from_popup_menu(lst, title=None, default=None):
    """Shows the list as a pop-up menu and returns the selected item as a string or None"""
    w = QMenu()
    _title = title if title is not None else 'Select'
    #w.setTitle(_title)
    atitle = w.addAction(_title)
    atitle.setDisabled(True)
    asep = w.addSeparator()
    for name in lst : 
       action = w.addAction(name)
       if name == default :
           #w.setDefaultAction(action)
           w.setActiveAction(action)
    item = w.exec_(QCursor.pos())
    return None if item is None else str(item.text()) # str(QString)

#------------------------------

def select_color(colini=Qt.blue, parent=None):
    """Select color using QColorDialog"""
    qcd = QColorDialog
    w = qcd(colini, parent)
    w.setOptions(qcd.ShowAlphaChannel)# | qcd.DontUseNativeDialog | qcd.NoButtons
    res = w.exec_()
    color=w.selectedColor()
    #color = QColorDialog.getColor()
    return None if color is None else color # QColor or None

#------------------------------

def change_check_box_list_in_popup_menu(list, win_title='Set check boxes'):
    """Shows the list of check-boxes as a dialog pop-up menu and returns the (un)changed list"""
    if list is None : return 0

    from psana.graphqt.QWPopupCheckList import QWPopupCheckList

    popupMenu = QWPopupCheckList(None, list, win_title)
    #popupMenu.move(QPoint(50,50))
    popupMenu.move(QCursor.pos())
    response = popupMenu.exec_()

    if   response == QDialog.Accepted :
        #logger.debug('New checkbox list is accepted', __name__)         
        return 1
    elif response == QDialog.Rejected :
        #logger.debug('Will use old checkbox list', __name__)
        return 0
    else                                    :
        #logger.error('Unknown response...', __name__)
        return 2

#------------------------------

def change_check_box_dict_in_popup_menu(dict, win_title='Set check boxes'):
    """Shows the dict of check-boxes as a dialog pop-up menu and returns the (un)changed dict"""
    if dict is None : return 0

    from psana.graphqt.QWPopupCheckDict import QWPopupCheckDict

    popupMenu = QWPopupCheckDict(None, dict, win_title)
    #popupMenu.move(QPoint(50,50))
    popupMenu.move(QCursor.pos())
    response = popupMenu.exec_()

    if   response == QDialog.Accepted :
        #logger.debug('New checkbox dict is accepted', __name__)         
        return 1
    elif response == QDialog.Rejected :
        #logger.debug('Will use old checkbox dict', __name__)
        return 0
    else                                    :
        #logger.error('Unknown response...', __name__)
        return 2

#------------------------------

def select_radio_button_in_popup_menu(dict_of_pars, win_title='Select option', do_confirm=False):
    """Popup GUI to select radio button from the list:  dict_of_pars = {'checked':'radio1', 'list':['radio0', 'radio1', 'radio2']}
    """
    from psana.graphqt.QWPopupRadioList import QWPopupRadioList

    popupMenu = QWPopupRadioList(None, dict_of_pars, win_title, do_confirm)
    #popupMenu.move(QPoint(50,50))
    popupMenu.move(QCursor.pos()-QPoint(100,100))
    return popupMenu.exec_() # QDialog.Accepted or QDialog.Rejected

#------------------------------

def print_rect(r, cmt='') :
    x, y, w, h = r.x(), r.y(), r.width(), r.height()
    L, R, T, B = r.left(), r.right(), r.top(), r.bottom()
    print('%s x=%8.2f  y=%8.2f  w=%8.2f  h=%8.2f' % (cmt, x, y, w, h))
    print('%s L=%8.2f  B=%8.2f  R=%8.2f  T=%8.2f' % (len(cmt)*' ', L, B, R, T))

#------------------------------

def get_save_fname_through_dialog_box(parent, path0, title, filter='*.txt'):       

    path, fext = QFileDialog.getSaveFileName(parent,
                                             caption   = title,
                                             directory = path0,
                                             filter    = filter
                                             )
    if path == '' :
        #logger.debug('Saving is cancelled.', 'get_save_fname_through_dialog_box')
        return None
    #logger.info('Output file: ' + path, 'get_save_fname_through_dialog_box')
    return path

#------------------------------

def get_open_fname_through_dialog_box(parent, path0, title, filter='*.txt'):       

    path, fext = QFileDialog.getOpenFileName(parent, title, path0, filter=filter)

    #print('XXX: get_open_fname_through_dialog_box path =', path)
    #print('XXX: get_open_fname_through_dialog_box fext =', fext)

    dname, fname = os.path.split(path)
    if dname == '' or fname == '' :
        #logger.info('Input directiry name or file name is empty... keep file path unchanged...')
        #print 'Input directiry name or file name is empty... keep file path unchanged...'
        return None
    #logger.info('Input file: ' + path, 'get_open_fname_through_dialog_box') 
    #print 'Input file: ' + path
    return path

#------------------------------

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

        #if   clicked == QtGui.QMessageBox.Save :
        #    logger.info('Saving is requested', __name__)
        #elif clicked == QtGui.QMessageBox.Discard :
        #    logger.info('Discard is requested', __name__)
        #else :
        #    logger.info('Cancel is requested', __name__)
        #return clicked

        #logger.info('You acknowkeged that saw the message:\n' + text, 'confirm_dialog_box')
        return

#------------------------------

def confirm_or_cancel_dialog_box(parent=None, text='Please confirm or cancel', title='Confirm or cancel') :
        """Pop-up MODAL box for confirmation"""

        mesbox = QMessageBox(parent, windowTitle=title,
                                           text=text,
                                           standardButtons=QMessageBox.Ok | QMessageBox.Cancel)
               #standardButtons=QtGui.QMessageBox.Save | QtGui.QMessageBox.Discard | QtGui.QMessageBox.Cancel)
        mesbox.setDefaultButton(QMessageBox.Ok)
        #mesbox.setMinimumSize(400, 200)
        #style = "background-color: rgb(255, 200, 220); color: rgb(0, 0, 100);" # Pinkish
        #style = "background-color: rgb(255, 255, 220); color: rgb(0, 0, 0);" # Yellowish
        #mesbox.setStyleSheet (style)

        clicked = mesbox.exec_() # DISPLAYS THE QMessageBox HERE

        if   clicked == QMessageBox.Ok     : return True
        elif clicked == QMessageBox.Cancel : return False
        else                               : return False

#------------------------------

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

#------------------------------

def widget_from_layout(l) :
    w = QWigget()
    w.setLayout(l)
    return w

#------------------------------

def layout_from_widget(w, layout=QVBoxLayout) :
    l = layout()
    l.addWidget(w)
    return l

#------------------------------
#------------------------------
 
if __name__ == "__main__" :

  def test(tname) :

    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)

    if tname == '0':
        instrs = ['SXR', 'AMO', 'XPP', 'CXI', 'MEC']
        resp = select_item_from_popup_menu(instrs, title='Select INS', default='AMO') 
        print('Selected:', resp)

    elif tname == '1':
        list_of_cbox = [['VAR1', True], ['VAR2', False], ['VAR3', False], ['VAR4', False], ['VAR5', False]]
        resp = change_check_box_list_in_popup_menu(list_of_cbox, win_title='Select vars(s)')
        for (var,stat) in list_of_cbox : print(var, stat)
        print('resp:', resp)
        
    elif tname == '2': 
        dict_of_pars = {'checked':'radio1', 'list':['radio0', 'radio1', 'radio2']}
        resp = select_radio_button_in_popup_menu(dict_of_pars, win_title='Select vars(s)', do_confirm=True)
        for (k,v) in dict_of_pars.items() : print(k, v)
        print('resp:', resp)

    elif tname == '3':
        parent=None; path0='./'; title='get_save_fname_through_dialog_box'
        resp = get_save_fname_through_dialog_box(parent, path0, title, filter='*.txt')
        print('Response:', resp)

    elif tname == '4': 
        parent=None; path0='./'; title='get_open_fname_through_dialog_box'
        resp = get_open_fname_through_dialog_box(parent, path0, title, filter='*.txt')
        print('Response:', resp)

    elif tname == '5': 
        resp = confirm_dialog_box(parent=None, text='Confirm that you aware!', title='Acknowledge')
        print('Response:', resp)

    elif tname == '6': 
        resp = confirm_or_cancel_dialog_box(parent=None, text='Confirm or cancel', title='Confirm or cancel') 
        print('Response:', resp)

    elif tname == '7': 
        from time import sleep
        resp = help_dialog_box(parent=None, text='Help message goes here', title='Help')
        print('Response:', resp)
        sleep(3)
        del resp

    elif tname == '8': 
        resp = select_color(colini=Qt.blue, parent=None)
        print('Response:', resp)

    elif tname == '9':
        dict_of_cbox = {'VAR1':True, 'VAR2':False, 'VAR3':False, 'VAR4':False, 'VAR5':False}
        resp = change_check_box_dict_in_popup_menu(dict_of_cbox, win_title='Select vars(s)')
        for (var,stat) in dict_of_cbox.items() : print(var, stat)
        print('resp:', resp)
        
    else :
        print('Sorry, not-implemented test "%s"' % tname)

    del app

#------------------------------

if __name__ == "__main__" :
    import sys; global sys
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print(50*'_', '\nTest %s' % tname)
    test(tname)
    sys.exit('End of test %s' % tname)

#------------------------------
