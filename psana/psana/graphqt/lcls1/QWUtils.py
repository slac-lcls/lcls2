#------------------------------
"""
@version $Id: QWUtils.py 13133 2017-02-08 01:05:23Z dubrovin@SLAC.STANFORD.EDU $

@author Mikhail S. Dubrovin
"""
#------------------------------

#import os
#import sys
from PyQt4 import QtGui, QtCore

#------------------------------

def selectFromListInPopupMenu(list):
    """Shows the list as a pop-up menu and returns the selected item as a string or None"""

    if list is None : return None
    
    popupMenu = QtGui.QMenu()
    for item in list :
        popupMenu.addAction(item)

    item_selected = popupMenu.exec_(QtGui.QCursor.pos())

    if item_selected is None : return None
    else                     : return str(item_selected.text()) # QString -> str

#------------------------------

def changeCheckBoxListInPopupMenu(list, win_title='Set check boxes'):
    """Shows the list of check-boxes as a dialog pop-up menu and returns the (un)changed list"""
    if list is None : return 0

    from CalibManager.GUIPopupCheckList import GUIPopupCheckList

    popupMenu = GUIPopupCheckList(None, list, win_title)
    #popupMenu.move(QtCore.QPoint(50,50))
    popupMenu.move(QtGui.QCursor.pos())
    response = popupMenu.exec_()

    if   response == QtGui.QDialog.Accepted :
        #logger.debug('New checkbox list is accepted', __name__)         
        return 1
    elif response == QtGui.QDialog.Rejected :
        #logger.debug('Will use old checkbox list', __name__)
        return 0
    else                                    :
        #logger.error('Unknown response...', __name__)
        return 2

#------------------------------

def selectRadioButtonInPopupMenu(dict_of_pars, win_title='Select option', do_confirm=False):
    """Popup GUI to select radio button from the list:  dict_of_pars = {'checked':'radio1', 'list':['radio0', 'radio1', 'radio2']}
    """
    from CalibManager.GUIPopupRadioList import GUIPopupRadioList

    popupMenu = GUIPopupRadioList(None, dict_of_pars, win_title, do_confirm)
    #popupMenu.move(QtCore.QPoint(50,50))
    popupMenu.move(QtGui.QCursor.pos()-QtCore.QPoint(100,100))
    return popupMenu.exec_() # QtGui.QDialog.Accepted or QtGui.QDialog.Rejected

#------------------------------
#------------------------------
#------------------------------
 
def test_all(tname) :

    app = QtGui.QApplication(sys.argv)

    if tname == '0':
        instrs = ['SXR', 'AMO', 'XPP', 'CXI', 'MEC']
        resp = selectFromListInPopupMenu(instrs) 
        print 'Selected:', resp

    elif tname == '1':
        list_of_cbox = [['VAR1', True], ['VAR2', False], ['VAR3', False], ['VAR4', False], ['VAR5', False]]
        resp = changeCheckBoxListInPopupMenu(list_of_cbox, win_title='Select vars(s)')
        for (var,stat) in list_of_cbox : print var, stat
        print 'resp:', resp
        
    elif tname == '2': 
        dict_of_pars = {'checked':'radio1', 'list':['radio0', 'radio1', 'radio2']}
        resp = selectRadioButtonInPopupMenu(dict_of_pars, win_title='Select vars(s)', do_confirm=True)
        for (k,v) in dict_of_pars.iteritems() : print k, v
        print 'resp:', resp

    else :
        print 'Sorry, not-implemented test "%s"' % tname

#------------------------------

if __name__ == "__main__" :
    import sys; global sys
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print 50*'_', '\nTest %s' % tname
    test_all(tname)
    sys.exit('End of test %s' % tname)

#------------------------------
