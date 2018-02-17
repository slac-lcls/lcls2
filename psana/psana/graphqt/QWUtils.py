#------------------------------
"""
:py:class:`QWUtils` - Popup GUIs
================================

Usage::

    # Import
    from psana.graphqt.QWUtils import QWUtils

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

from PyQt5 import QtWidgets
from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QCursor

#------------------------------

def selectFromListInPopupMenu(list):
    """Shows the list as a pop-up menu and returns the selected item as a string or None"""

    if list is None : return None
    
    popupMenu = QtWidgets.QMenu()
    for item in list :
        popupMenu.addAction(item)

    item_selected = popupMenu.exec_(QCursor.pos())

    if item_selected is None : return None
    else                     : return str(item_selected.text()) # QString -> str

#------------------------------

def changeCheckBoxListInPopupMenu(list, win_title='Set check boxes'):
    """Shows the list of check-boxes as a dialog pop-up menu and returns the (un)changed list"""
    if list is None : return 0

    from psana.graphqt.QWPopupCheckList import QWPopupCheckList

    popupMenu = QWPopupCheckList(None, list, win_title)
    #popupMenu.move(QPoint(50,50))
    popupMenu.move(QCursor.pos())
    response = popupMenu.exec_()

    if   response == QtWidgets.QDialog.Accepted :
        #logger.debug('New checkbox list is accepted', __name__)         
        return 1
    elif response == QtWidgets.QDialog.Rejected :
        #logger.debug('Will use old checkbox list', __name__)
        return 0
    else                                    :
        #logger.error('Unknown response...', __name__)
        return 2

#------------------------------

def selectRadioButtonInPopupMenu(dict_of_pars, win_title='Select option', do_confirm=False):
    """Popup GUI to select radio button from the list:  dict_of_pars = {'checked':'radio1', 'list':['radio0', 'radio1', 'radio2']}
    """
    from psana.graphqt.QWPopupRadioList import QWPopupRadioList

    popupMenu = QWPopupRadioList(None, dict_of_pars, win_title, do_confirm)
    #popupMenu.move(QPoint(50,50))
    popupMenu.move(QCursor.pos()-QPoint(100,100))
    return popupMenu.exec_() # QtWidgets.QDialog.Accepted or QtWidgets.QDialog.Rejected

#------------------------------

def print_rect(r, cmt='') :
    x, y, w, h = r.x(), r.y(), r.width(), r.height()
    L, R, T, B = r.left(), r.right(), r.top(), r.bottom()
    print('%s x=%8.2f  y=%8.2f  w=%8.2f  h=%8.2f' % (cmt, x, y, w, h))
    print('%s L=%8.2f  B=%8.2f  R=%8.2f  T=%8.2f' % (len(cmt)*' ', L, B, R, T))

#------------------------------
#------------------------------
#------------------------------
 
def test_all(tname) :

    app = QtWidgets.QApplication(sys.argv)

    if tname == '0':
        instrs = ['SXR', 'AMO', 'XPP', 'CXI', 'MEC']
        resp = selectFromListInPopupMenu(instrs) 
        print('Selected:', resp)

    elif tname == '1':
        list_of_cbox = [['VAR1', True], ['VAR2', False], ['VAR3', False], ['VAR4', False], ['VAR5', False]]
        resp = changeCheckBoxListInPopupMenu(list_of_cbox, win_title='Select vars(s)')
        for (var,stat) in list_of_cbox : print(var, stat)
        print('resp:', resp)
        
    elif tname == '2': 
        dict_of_pars = {'checked':'radio1', 'list':['radio0', 'radio1', 'radio2']}
        resp = selectRadioButtonInPopupMenu(dict_of_pars, win_title='Select vars(s)', do_confirm=True)
        for (k,v) in dict_of_pars.items() : print(k, v)
        print('resp:', resp)

    else :
        print('Sorry, not-implemented test "%s"' % tname)

#------------------------------

if __name__ == "__main__" :
    import sys; global sys
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print(50*'_', '\nTest %s' % tname)
    test_all(tname)
    sys.exit('End of test %s' % tname)

#------------------------------
