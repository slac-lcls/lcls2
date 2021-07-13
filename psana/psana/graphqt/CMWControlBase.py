
"""Class :py:class:`CMWControlBase` is a QWidget base class for control buttons
===============================================================================

Usage ::

    # Run test: python lcls2/psana/psana/graphqt/CMWControlBase.py

    from psana.graphqt.CMWControlBase import CMWControlBase
    w = CMWControlBase()

Created on 2021-06-16 by Mikhail Dubrovin
"""

#import logging
#logger = logging.getLogger(__name__)

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton
from psana.graphqt.CMConfigParameters import cp
from psana.graphqt.Styles import style


class CMWControlBase(QWidget):
    """QWidget - base class for control buttons"""

    def __init__(self, **kwargs):

        parent = kwargs.get('parent',None)
        QWidget.__init__(self, parent)

        self.but_tabs = QPushButton('Tabs %s' % cp.char_expand)
        self.but_tabs.clicked.connect(self.on_but_tabs)

        if __name__ == "__main__":
            self.box1 = QVBoxLayout() 
            #self.box1.addSpacing(20)
            self.box1.addStretch(1) 
            self.box1.addWidget(self.but_tabs)
            #self.box1.addLayout(self.grid)
            self.setLayout(self.box1)
            #self.set_buttons_visiable()

            self.set_tool_tips()
            self.set_style()


    def set_tool_tips(self):
        self.but_tabs.setToolTip('Show/hide tabs')
 

    def set_style(self):
        self.but_tabs.setStyleSheet(style.styleButtonGood)
        self.but_tabs.setFixedWidth(55)


    def on_but_tabs(self):
        #logger.debug('on_but_tabs')
        self.view_hide_tabs()


    def view_hide_tabs(self):
        wtabs = cp.cmwmaintabs
        if wtabs is None: return
        is_visible = wtabs.tab_bar_is_visible()
        self.but_tabs.setText('Tabs %s'%cp.char_shrink if is_visible else 'Tabs %s'%cp.char_expand)
        wtabs.set_tabs_visible(not is_visible)


    def but_tabs_is_visible(self, isvisible=True):
        self.but_tabs.setVisible(isvisible)


if __name__ == "__main__":
    import os
    import sys
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1' #export LIBGL_ALWAYS_INDIRECT=1
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = CMWControlBase()
    w.setGeometry(100, 50, 200, 50)
    w.setWindowTitle('Control Base')
    w.show()
    app.exec_()
    del w
    del app

# EOF
