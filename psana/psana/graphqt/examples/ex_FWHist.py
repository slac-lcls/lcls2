
from psana.graphqt.FWHist import *

class TestFWHist(FWHist):

    def add_test(self):

        r = self.scene.sceneRect()
        w,h = r.width(), r.height()
        v = self.view
        sv = 1 if v._origin_u else -1
        sh = 1 if v._origin_l else -1

        self.pbl   = r.bottomLeft()
        self.pbr   = r.bottomRight()
        self.ptl   = r.topLeft()
        self.ptr   = r.topRight()
        self.ptc   = (self.ptl + self.ptr)/2
        self.pbc   = (self.pbl + self.pbr)/2
        self.plc   = (self.pbl + self.ptl)/2
        self.prc   = (self.pbr + self.ptr)/2
        self.pcc   = (self.pbc + self.ptc)/2

        if self.path_item is not None: self.scene.removeItem(self.path_item)

        self.path = QPainterPath(self.pbl)
        #self.path.closeSubpath()
        self.path.moveTo(self.pbl)
        self.path.lineTo(self.ptc)
        self.path.lineTo(self.pcc)
        self.path.lineTo(self.prc)
        self.path.lineTo(self.pbl)

        # add path with hist lines to scene
        self.lst_of_items=[]

        self.path_item = self.scene.addPath(self.path, self.pen, self.brush)
        self.path_item.setZValue(self.zvalue)
        self.lst_of_items.append(self.path_item)

        print('test path_item is created')


def test_histogram():
    import psana.pyalgos.generic.NDArrGenerators as ag
    from psana.pyalgos.generic.HBins import HBins
    nbins = 1000
    arr = ag.random_standard((nbins,), mu=50, sigma=10, dtype=ag.np.float64)
    hb = HBins((0,nbins), nbins=nbins)
    hb.set_bin_data(arr, dtype=ag.np.float64)
    return hb


if __name__ == "__main__":

  import os
  import sys

  def test_fwhist(tname):

    from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsView
    from PyQt5.QtCore import QRectF

    app = QApplication(sys.argv)
    rs = QRectF(0, 0, 100, 100) # scene rect
    ro = QRectF(-2, -1, 4, 2)   # origin rect for mark only
    s = QGraphicsScene(rs)
    v = QGraphicsView(s)
    v.setGeometry(20, 20, 600, 600)
    v._origin_u = True
    v._origin_l = True

    print('screenGeometry():', app.desktop().screenGeometry())
    print('scene rect=', s.sceneRect())

    v.fitInView(rs, Qt.KeepAspectRatio) # Qt.IgnoreAspectRatio Qt.KeepAspectRatioByExpanding Qt.KeepAspectRatio

    # mark scene and origin by colored rects
    s.addRect(rs, pen=QPen(Qt.black, 0, Qt.SolidLine), brush=QBrush(Qt.yellow))
    s.addRect(ro, pen=QPen(Qt.black, 0, Qt.SolidLine), brush=QBrush(Qt.red))

    if   tname ==  '0': h = TestFWHist(v, hbins=test_histogram())
    elif tname ==  '1': h = TestFWHist(v, hbins=None)
    elif tname ==  '2': h = TestFWHist(v, hbins=test_histogram(), orient='V')
    elif tname ==  '3': h = TestFWHist(v, hbins=test_histogram(), orient='V', brush=QBrush(Qt.green), color=Qt.blue)
    elif tname ==  '4': h = TestFWHist(v, hbins=test_histogram(), orient='H', brush=QBrush(Qt.green), pen=QPen(Qt.magenta, 2, Qt.SolidLine))
    else:
        print('test %s is not implemented' % tname)
        return

    v.setWindowTitle("TestFWHist test")
    v.setContentsMargins(0,0,0,0)
    v.show()
    app.exec_()

    # prevent crash on exit...
    del h, v, s, app


if __name__ == "__main__":

    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1' #export LIBGL_ALWAYS_INDIRECT=1
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print(50*'_', '\nTest %s' % tname)
    test_fwhist(tname)
    #sys.exit('End of Test %s' % tname)

# EOF
