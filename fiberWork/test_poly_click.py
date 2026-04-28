import sys
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from PyQt5.QtGui import QPixmap, QPolygon, QPainter, QPen, QBrush
from PyQt5.QtCore import QPoint
from PyQt5.QtCore import Qt as Qt

class TextEditDemo(QWidget):
    def __init__(self, parent=None):
        super().__init__()

        
        view = QGraphicsView()
        
        # create the image

        self.mousePressEvent = self.img_click
        # create the polygons
        first_polygon  = QPolygon() << QPoint(0, 0) << QPoint(100, 0) << QPoint(100, 100) << QPoint(0, 100)
        second_polygon = QPolygon() << QPoint(200, 200) << QPoint(300, 200) << QPoint(300, 300) << QPoint(200, 300)

        #points = QPolygon([ QPoint(10,10), QPoint(10,100),
        #     QPoint(100,10), QPoint(100,100)])


        # create a dictionary containing the name of the area, the polygon and the function to be called when
        # the polygon is clicked
        self.clickable_areas = {
            "first": {
                "polygon": first_polygon,
                "func": self.func1
            },
            "second": {
                "polygon": second_polygon,
                "func": self.func2
            }
        }

    def img_click(self, event):
        # get the position of the click
        pos = event.pos()

        # iterate over all polygons
        for area in self.clickable_areas:
            # if the point is inside one of the polygons, call the function associated with that polygon
            if self.clickable_areas[area]["polygon"].containsPoint(pos, Qt.FillRule.OddEvenFill):
                self.clickable_areas[area]["func"]()
                return
            self.func3()


    # the functions to be called when specific polygons are clicked
    def func1(self):
        print("first polygon clicked!")

    def func2(self):
        print("second polygon clicked!")

    def func3(self):
        print("no polygon was clicked")

    def paintEvent(self, event):
        painter = QPainter(self)
        pixmap = QPixmap("/cds/home/m/melchior/images/Screenshot 2025-08-14 at 9.07.07 AM.png")
        self.max_size=(500,50)
        scaled_pixmap = pixmap.scaled(
                    self.max_size[0], self.max_size[1], 
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
        painter.drawPixmap(self.rect(), scaled_pixmap)
        pen = QPen(Qt.red, 5, Qt.SolidLine)
        painter.setPen(pen)
        for area in self.clickable_areas:
            painter.drawPolygon(self.clickable_areas[area]["polygon"])
        painter.end()
                # connect the mouse press event on the image to the img_click function
        



def main():
    app = QApplication(sys.argv)
    win = TextEditDemo()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
