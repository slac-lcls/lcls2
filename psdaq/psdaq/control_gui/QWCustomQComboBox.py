#!/usr/bin/env python

"""CustomQComboBox extension of QComboBox for control_gui
   - get rid of responce on Qt.Key_Up, Key_Dow and mouth wheelEvent.
@date 2025-05-15
@author Mikhail D
"""

import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QComboBox

class CustomQComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)

    def keyPressEvent(self, event):
        #print('event.key():', event.key())
        if event.key() in (Qt.Key_Up, Qt.Key_Down):
            event.ignore()  # Ignore up and down arrow keys
        else:
            super().keyPressEvent(event) # Default behavior for other keys

    def wheelEvent(self, event):
        event.ignore()
        #super().wheelEvent(event) # Default behavior
        #if event.angleDelta().y() in (120, -120):
        #    event.ignore()  # Ignore up and down arrow keys
        #else:
        #    super().keyPressEvent(event) # Default behavior for other keys


if __name__ == "__main__":
    """test CustomQComboBox for control_gui"""
    from PyQt5.QtWidgets import QMainWindow, QApplication
    from PyQt5.QtGui import QCursor

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            combobox = CustomQComboBox()
            combobox.addItems(['One', 'Two', 'Three', 'Four'])
            # Connect signals to the methods.
            combobox.activated.connect(self.activated)
            combobox.currentTextChanged.connect(self.text_changed)
            combobox.currentIndexChanged.connect(self.index_changed)

            self.setCentralWidget(combobox)

        def activated(Self, index):
            print("Activated index:", index)

        def text_changed(self, s):
            print("Text changed:", s)

        def index_changed(self, index):
            print("Index changed", index)

    app = QApplication(sys.argv)
    w = MainWindow()
    w.move(QCursor.pos())
    w.show()
    app.exec_()

# EOF
