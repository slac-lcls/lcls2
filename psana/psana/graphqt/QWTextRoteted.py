import sys

from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QWidget

class QWTextRotated(QWidget):
    def __init__(self, text='', angle=-90, **kwa):
        QWidget.__init__(self)
        self.text = text
        self.angle = angle
        self.kwa = kwa

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setPen(self.kwa.get('pen',QtGui.QPen(QtCore.Qt.black, 0, QtCore.Qt.SolidLine)))
        painter.setBrush(self.kwa.get('brush', QtGui.QBrush()))
        painter.translate(self.kwa.get('translate', QtCore.QPoint(50, 50))) # increment for the next point by default
        painter.setFont(self.kwa.get('font', QtGui.QFont())) #QFont('times',22)
        painter.rotate(self.angle)
        painter.drawText(self.kwa.get('point', QtCore.QPoint(0, 0)), self.text)
        painter.end()

if __name__ == '__main__':

    from PyQt5.QtWidgets import QApplication, QHBoxLayout
    class Example(QWidget):
      def __init__(self):
        QWidget.__init__(self)
        text1 = QWTextRotated('text 1', 10)
        text2 = QWTextRotated('text 2', 45)
        text3 = QWTextRotated('text 3',-45)
        hBoxLayout = QHBoxLayout()
        hBoxLayout.addWidget(text1)
        hBoxLayout.addWidget(text2)
        hBoxLayout.addWidget(text3)
        self.setLayout(hBoxLayout)
        self.setGeometry(110, 10, 600, 400)
        self.show()

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())

#EOF
