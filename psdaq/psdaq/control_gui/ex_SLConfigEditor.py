#####!/usr/bin/env python
#--------------------
"""
:py:class:`SLConfigEditor` - widget for StandAlone (SL) configuration editor
============================================================================

Usage::
    - python lcls2/psdaq/psdaq/control_gui/ex_SLConfigEditor.py

See:
    - :py:class:`CGWConfigEditor`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2/psdaq/psdaq/control_gui>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2020-01-16 by Mikhail Dubrovin
"""
#--------------------

import logging
logger = logging.getLogger('SLConfigEditor')

from psdaq.control_gui.CGWConfigEditor import CGWConfigEditor, str_json

#--------------------

class SLConfigEditor(CGWConfigEditor) :
    """Standalone Configuration Editor
    """
    def __init__(self, parent=None, dictj=None):
        CGWConfigEditor.__init__(self, parent, dictj=None)

#--------------------
 
    def on_but_apply(self):
        logger.debug('on_but_apply')
        dj = self.get_content()
        sj = str_json(dj)
        logger.info('on_but_apply jason/dict:\n%s' % sj)
        logger.warning("TBD - DO SOMETHING HERE WITH YOUR JSON DICT")

        #if cp.cgwmainconfiguration is None :
        #    logger.warning("parent (ctrl) is None - changes can't be applied to DB")
        #    return
        #else :
        #    cp.cgwmainconfiguration.save_dictj_in_db(dj, msg='CGWConfigEditor: ')
            
#--------------------
 
    def closeEvent(self, e):
        CGWConfigEditor.closeEvent(self, e)
        logger.debug('closeEvent')
        if self.help_box is not None : self.help_box.close()

        #if cp.cgwmainconfiguration is not None :
        #   cp.cgwmainconfiguration.w_edit = None

#--------------------

if __name__ == "__main__" :

    logging.basicConfig(format='%(levelname)s %(name)s: %(message)s', level=logging.DEBUG)

    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = SLConfigEditor(dictj=None) # if dictj is None it is loaded from test file
    w.show()
    app.exec_()

#--------------------
