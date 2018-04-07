#------------------------------
"""
:py:class:`QWIcons` - singleton access to icons
================================================

Usage::

    # Test: python lcls2/psana/psana/graphqt/QWIcons.py

    # Import
    from psana.graphqt.QWIcons import icon
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # Get QIcon objects
    icon.set_icons()
    icon1 = icon.icon_exit
    icon2 = icon.icon_home

See:
    - :py:class:`QWIcons`
    - `lcls2 on github <https://github.com/slac-lcls/lcls2>`_.

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2016-11-22 by Mikhail Dubrovin
Adopted for LCLS2 on 2018-02-15
"""
#------------------------------

import os
from PyQt5.QtGui import QIcon

#------------------------------

class QWIcons() :
    """A singleton storage of icons with caching.
    """
    def __init__(self) :
        self._name = self.__class__.__name__
        self.icons_are_loaded = False

#------------------------------

    def path_to_icons(self) :
        _ROOT = os.path.abspath(os.path.dirname(__file__))
        #path_icon = 'psana/psana/graphqt/data/icons'
        path_icon = '%s/data/icons' % _ROOT
        #print('XXX set_icons :path_icon', path_icon)
        return path_icon

#------------------------------
#    def path_to_icons_v1(self) :
#        import pkgutil
#        path_icon = pkgutil.get_data('graphqt', 'data/icons/contents.png')
#        print('XXX set_icons :path_icon', path_icon)
#        print('XXX set_icons :__name__', __name__)
#        print('XXX set_icons :__file__', __file__)
#        return path_icon
#------------------------------

    def set_icons(self) :
        """QIcon SHOULD BE CALLED AFTER QApplication"""

        if self.icons_are_loaded : return
        self.icons_are_loaded = True

        path_icon = self.path_to_icons()

        self.path_icon_contents      = '%s/contents.png' % path_icon     
        self.path_icon_mail_forward  = '%s/mail-forward.png' % path_icon 
        self.path_icon_button_ok     = '%s/button_ok.png' % path_icon    
        self.path_icon_button_cancel = '%s/button_cancel.png' % path_icon
        self.path_icon_exit          = '%s/exit.png' % path_icon         
        self.path_icon_home          = '%s/home.png' % path_icon         
        self.path_icon_redo          = '%s/redo.png' % path_icon         
        self.path_icon_undo          = '%s/undo.png' % path_icon         
        self.path_icon_reload        = '%s/reload.png' % path_icon       
        self.path_icon_save          = '%s/save.png' % path_icon         
        self.path_icon_save_cfg      = '%s/fileexport.png' % path_icon   
        self.path_icon_edit          = '%s/edit.png' % path_icon         
        self.path_icon_browser       = '%s/fileopen.png' % path_icon     
        self.path_icon_monitor       = '%s/icon-monitor.png' % path_icon 
        self.path_icon_unknown       = '%s/icon-unknown.png' % path_icon 
        self.path_icon_plus          = '%s/icon-plus.png' % path_icon    
        self.path_icon_minus         = '%s/icon-minus.png' % path_icon   
        self.path_icon_logviewer     = '%s/logviewer.png' % path_icon    
        self.path_icon_lock          = '%s/locked-icon.png' % path_icon  
        self.path_icon_unlock        = '%s/unlocked-icon.png' % path_icon
        self.path_icon_convert       = '%s/icon-convert.png' % path_icon 
        self.path_icon_table         = '%s/table.gif' % path_icon        
        self.path_icon_folder_open   = '%s/folder_open.gif' % path_icon  
        self.path_icon_folder_closed = '%s/folder_closed.gif' % path_icon
        self.path_icon_expcheck      = '%s/folder_open_checked.png' % path_icon
 
        self.icon_contents      = QIcon(self.path_icon_contents     )
        self.icon_mail_forward  = QIcon(self.path_icon_mail_forward )
        self.icon_button_ok     = QIcon(self.path_icon_button_ok    )
        self.icon_button_cancel = QIcon(self.path_icon_button_cancel)
        self.icon_exit          = QIcon(self.path_icon_exit         )
        self.icon_home          = QIcon(self.path_icon_home         )
        self.icon_redo          = QIcon(self.path_icon_redo         )
        self.icon_undo          = QIcon(self.path_icon_undo         )
        self.icon_reload        = QIcon(self.path_icon_reload       )
        self.icon_save          = QIcon(self.path_icon_save         )
        self.icon_save_cfg      = QIcon(self.path_icon_save_cfg     )
        self.icon_edit          = QIcon(self.path_icon_edit         )
        self.icon_browser       = QIcon(self.path_icon_browser      )
        self.icon_monitor       = QIcon(self.path_icon_monitor      )
        self.icon_unknown       = QIcon(self.path_icon_unknown      )
        self.icon_plus          = QIcon(self.path_icon_plus         )
        self.icon_minus         = QIcon(self.path_icon_minus        )
        self.icon_logviewer     = QIcon(self.path_icon_logviewer    )
        self.icon_lock          = QIcon(self.path_icon_lock         )
        self.icon_unlock        = QIcon(self.path_icon_unlock       )
        self.icon_convert       = QIcon(self.path_icon_convert      )
        self.icon_table         = QIcon(self.path_icon_table        )
        self.icon_folder_open   = QIcon(self.path_icon_folder_open  )
        self.icon_folder_closed = QIcon(self.path_icon_folder_closed)
        self.icon_expcheck      = QIcon(self.path_icon_expcheck     )

        self.icon_data          = self.icon_table
        self.icon_apply         = self.icon_button_ok
        self.icon_reset         = self.icon_undo
        self.icon_retreve       = self.icon_redo
        self.icon_expand        = self.icon_folder_open
        self.icon_collapse      = self.icon_folder_closed
        self.icon_print         = self.icon_contents
 
#------------------------------
        
icon = QWIcons()

#------------------------------

if __name__ == "__main__" :

  def test_QWIcons() :
    print('Icon pathes:')
    print(icon.path_icon_contents)
    print(icon.path_icon_mail_forward)
    print(icon.path_icon_button_ok)
    print(icon.path_icon_button_cancel)
    print(icon.path_icon_exit)
    print(icon.path_icon_home)
    print(icon.path_icon_redo)
    print(icon.path_icon_undo)  
    print(icon.path_icon_reload)
    print(icon.path_icon_save)
    print(icon.path_icon_save_cfg)
    print(icon.path_icon_edit)
    print(icon.path_icon_browser)
    print(icon.path_icon_monitor)
    print(icon.path_icon_unknown)
    print(icon.path_icon_plus)
    print(icon.path_icon_minus)
    print(icon.path_icon_logviewer)
    print(icon.path_icon_lock)
    print(icon.path_icon_unlock)
    print(icon.path_icon_convert)
    print(icon.path_icon_table)
    print(icon.path_icon_folder_open)
    print(icon.path_icon_folder_closed)

#------------------------------

if __name__ == "__main__" :
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    icon.set_icons()
    test_QWIcons()
    sys.exit(0)

#------------------------------
