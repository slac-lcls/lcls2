#------------------------------
"""
:py:class:`QWIcons` - singleton access to icons
================================================

Usage::

    # Test: python lcls2/psdaq/psdaq/control_gui/QWIcons.py

    # Import
    from psdaq.control_gui.QWIcons import icon
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

Copied from psana/psana/graphqt/QWIcons.py on 2019-01-29 by Mikhail Dubrovin
"""
#------------------------------

import os
import logging
logger = logging.getLogger(__name__)

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
        #logger.debug('XXX set_icons :path_icon', path_icon)
        return path_icon

#------------------------------
#    def path_to_icons_v1(self) :
#        import pkgutil
#        path_icon = pkgutil.get_data('graphqt', 'data/icons/contents.png')
#        logger.debug('XXX set_icons :path_icon', path_icon)
#        logger.debug('XXX set_icons :__name__', __name__)
#        logger.debug('XXX set_icons :__file__', __file__)
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

        self.path_icon_eject                  = '%s/media-eject.png'                                % path_icon
        self.path_icon_eject_sym              = '%s/media-eject-symbolic.symbolic.png'              % path_icon
        self.path_icon_playback_pause         = '%s/media-playback-pause.png'                       % path_icon
        self.path_icon_playback_pause_sym     = '%s/media-playback-pause-symbolic.symbolic.png'     % path_icon
        self.path_icon_playback_start         = '%s/media-playback-start.png'                       % path_icon
        self.path_icon_playback_start_rtl     = '%s/media-playback-start-rtl.png'                   % path_icon
        self.path_icon_playback_start_sym_rtl = '%s/media-playback-start-symbolic-rtl.symbolic.png' % path_icon
        self.path_icon_playback_start_sym     = '%s/media-playback-start-symbolic.symbolic.png'     % path_icon
        self.path_icon_playback_stop          = '%s/media-playback-stop.png'                        % path_icon
        self.path_icon_playback_stop_sym      = '%s/media-playback-stop-symbolic.symbolic.png'      % path_icon
        #self.path_icon_record_ring_red        = '%s/media-record-ring-red.png'                      % path_icon
        #self.path_icon_record_ring_black      = '%s/media-record-ring-black.png'                    % path_icon
        self.path_icon_seek_backward          = '%s/media-seek-backward.png'                        % path_icon
        self.path_icon_seek_backward_rtl      = '%s/media-seek-backward-rtl.png'                    % path_icon
        self.path_icon_seek_backward_sym_rtl  = '%s/media-seek-backward-symbolic-rtl.symbolic.png'  % path_icon
        self.path_icon_seek_backward_sym      = '%s/media-seek-backward-symbolic.symbolic.png'      % path_icon
        self.path_icon_seek_forward           = '%s/media-seek-forward.png'                         % path_icon
        self.path_icon_seek_forward_rtl       = '%s/media-seek-forward-rtl.png'                     % path_icon
        self.path_icon_seek_forward_sym_rtl   = '%s/media-seek-forward-symbolic-rtl.symbolic.png'   % path_icon
        self.path_icon_seek_forward_sym       = '%s/media-seek-forward-symbolic.symbolic.png'       % path_icon
        self.path_icon_skip_backward          = '%s/media-skip-backward.png'                        % path_icon
        self.path_icon_skip_backward_rtl      = '%s/media-skip-backward-rtl.png'                    % path_icon
        self.path_icon_skip_backward_sym_rtl  = '%s/media-skip-backward-symbolic-rtl.symbolic.png'  % path_icon
        self.path_icon_skip_backward_sym      = '%s/media-skip-backward-symbolic.symbolic.png'      % path_icon
        self.path_icon_skip_forward           = '%s/media-skip-forward.png'                         % path_icon
        self.path_icon_skip_forward_rtl       = '%s/media-skip-forward-rtl.png'                     % path_icon
        self.path_icon_skip_forward_sym_rtl   = '%s/media-skip-forward-symbolic-rtl.symbolic.png'   % path_icon
        self.path_icon_skip_forward_sym       = '%s/media-skip-forward-symbolic.symbolic.png'       % path_icon
        self.path_icon_view_subtitles_sym     = '%s/media-view-subtitles-symbolic.symbolic.png'     % path_icon
        self.path_icon_lcls                   = '%s/lcls.png'                                       % path_icon
        self.path_icon_wait                   = '%s/hourglass.png'                                  % path_icon
        #self.path_icon_circle_green           = '%s/circle-green-wide.png'                          % path_icon
        self.path_icon_checkbox_empty         = '%s/checkbox-empty.png'                             % path_icon
        self.path_icon_checkbox_red           = '%s/checkbox-checked-red.png'                       % path_icon
        self.path_icon_data_green             = '%s/icon-data-green.png'                            % path_icon
        self.path_icon_data_red_cross         = '%s/icon-data-red-cross.png'                        % path_icon
        #self.path_icon_checkbox_black         = '%s/checkbox-checked-black.png'                     % path_icon

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
        self.icon_lcls          = QIcon(self.path_icon_lcls         )
        self.icon_wait          = QIcon(self.path_icon_wait         )
        #self.icon_record_start  = QIcon(self.path_icon_record_ring_black)
        #self.icon_circle_green  = QIcon(self.path_icon_circle_green )
        #self.icon_ring_red      = QIcon(self.path_icon_record_ring_red)
        #self.icon_ring_black    = QIcon(self.path_icon_record_ring_black)
        self.icon_checkbox_empty= QIcon(self.path_icon_checkbox_empty)
        self.icon_checkbox_check= QIcon(self.path_icon_checkbox_red )

        self.icon_data_green    = QIcon(self.path_icon_data_green)
        self.icon_data_red_cross= QIcon(self.path_icon_data_red_cross)

        self.icon_data          = self.icon_table
        self.icon_apply         = self.icon_button_ok
        self.icon_reset         = self.icon_undo
        self.icon_retreve       = self.icon_redo
        self.icon_expand        = self.icon_folder_open
        self.icon_collapse      = self.icon_folder_closed
        self.icon_print         = self.icon_contents
        self.icon_record_stop   = self.icon_data_green
        self.icon_record_start  = self.icon_data_red_cross # icon_circle_green
        #self.icon_record_stop   = self.icon_checkbox_check
        #self.icon_record_start  = self.icon_checkbox_empty # icon_circle_green
        #self.icon_record_stop   = self.icon_ring_red
        #self.icon_record_start  = self.icon_circle_green

        self.icon_eject                  = QIcon(self.path_icon_eject                 )
        self.icon_eject_sym              = QIcon(self.path_icon_eject_sym             )
        self.icon_playback_pause         = QIcon(self.path_icon_playback_pause        )
        self.icon_playback_pause_sym     = QIcon(self.path_icon_playback_pause_sym    )
        self.icon_playback_start         = QIcon(self.path_icon_playback_start        )
        self.icon_playback_start_rtl     = QIcon(self.path_icon_playback_start_rtl    )
        self.icon_playback_start_sym_rtl = QIcon(self.path_icon_playback_start_sym_rtl)
        self.icon_playback_start_sym     = QIcon(self.path_icon_playback_start_sym    )
        self.icon_playback_stop          = QIcon(self.path_icon_playback_stop         )
        self.icon_playback_stop_sym      = QIcon(self.path_icon_playback_stop_sym     )

        self.icon_seek_backward          = QIcon(self.path_icon_seek_backward         )
        self.icon_seek_backward_rtl      = QIcon(self.path_icon_seek_backward_rtl     )
        self.icon_seek_backward_sym_rtl  = QIcon(self.path_icon_seek_backward_sym_rtl )
        self.icon_seek_backward_sym      = QIcon(self.path_icon_seek_backward_sym     )
        self.icon_seek_forward           = QIcon(self.path_icon_seek_forward          )
        self.icon_seek_forward_rtl       = QIcon(self.path_icon_seek_forward_rtl      )
        self.icon_seek_forward_sym_rtl   = QIcon(self.path_icon_seek_forward_sym_rtl  )
        self.icon_seek_forward_sym       = QIcon(self.path_icon_seek_forward_sym      )
        self.icon_skip_backward          = QIcon(self.path_icon_skip_backward         )
        self.icon_skip_backward_rtl      = QIcon(self.path_icon_skip_backward_rtl     )
        self.icon_skip_backward_sym_rtl  = QIcon(self.path_icon_skip_backward_sym_rtl )
        self.icon_skip_backward_sym      = QIcon(self.path_icon_skip_backward_sym     )
        self.icon_skip_forward           = QIcon(self.path_icon_skip_forward          )
        self.icon_skip_forward_rtl       = QIcon(self.path_icon_skip_forward_rtl      )
        self.icon_skip_forward_sym_rtl   = QIcon(self.path_icon_skip_forward_sym_rtl  )
        self.icon_skip_forward_sym       = QIcon(self.path_icon_skip_forward_sym      )
        self.icon_view_subtitles_sym     = QIcon(self.path_icon_view_subtitles_sym    )
 
#------------------------------
        
icon = QWIcons()

#------------------------------

if __name__ == "__main__" :

  def test_QWIcons() :
    logger.debug('Icon pathes:')
    logger.debug(icon.path_icon_contents)
    logger.debug(icon.path_icon_mail_forward)
    logger.debug(icon.path_icon_button_ok)
    logger.debug(icon.path_icon_button_cancel)
    logger.debug(icon.path_icon_exit)
    logger.debug(icon.path_icon_home)
    logger.debug(icon.path_icon_redo)
    logger.debug(icon.path_icon_undo)  
    logger.debug(icon.path_icon_reload)
    logger.debug(icon.path_icon_save)
    logger.debug(icon.path_icon_save_cfg)
    logger.debug(icon.path_icon_edit)
    logger.debug(icon.path_icon_browser)
    logger.debug(icon.path_icon_monitor)
    logger.debug(icon.path_icon_unknown)
    logger.debug(icon.path_icon_plus)
    logger.debug(icon.path_icon_minus)
    logger.debug(icon.path_icon_logviewer)
    logger.debug(icon.path_icon_lock)
    logger.debug(icon.path_icon_unlock)
    logger.debug(icon.path_icon_convert)
    logger.debug(icon.path_icon_table)
    logger.debug(icon.path_icon_folder_open)
    logger.debug(icon.path_icon_folder_closed)

#------------------------------

if __name__ == "__main__" :
    import sys
    from PyQt5.QtWidgets import QApplication
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    app = QApplication(sys.argv)
    icon.set_icons()
    test_QWIcons()
    sys.exit(0)

#------------------------------
