#!/usr/bin/env python

"""Class :py:class:`ex_GWViewImAx` - test
==========================================

Usage ::

    # Test: lcls2/psana/psana/graphqt/GWViewImAx.py

Created on 2023-05-11 by Mikhail Dubrovin
"""

from psana.graphqt.GWViewImAx import *
from psana.pyalgos.generic.NDArrGenerators import np, test_image, random_standard

#SCRNAME = sys.argv[0].split('/')[-1]
#USAGE = '\nUsage: %s' % SCRNAME

class TestGWViewImAx(GWViewImAx):

      def __init__(self, **kwa):
          GWViewImAx.__init__(self, **kwa)
          print(self.key_usage())

      def key_usage(self):
        return 'Keys:'\
               '\n  ESC - exit'\
               '\n  R - reset original size'\
               '\n  N - set new image'\
               '\n'

      def keyPressEvent(self, e):
        logger.debug('==  keyPressEvent key=%s' % e.key())
        if   e.key() == Qt.Key_Escape:
            print('Close app')
            self.close()

        elif e.key() == Qt.Key_R:
            r = self.wim.scene_rect()
            print('Reset original size wim.scene_rect: %s' % qu.info_rect_xywh(r))
            self.on_but_reset()
            #self.reset_scene_rect()  # it does not reset axes

        elif e.key() == Qt.Key_N:
            shran = np.maximum(random_standard(shape=(2,), mu=500, sigma=500, dtype=np.int32), (100,100))
            mu, sigma = random_standard(shape=(2,), mu=500, sigma=200, dtype=np.float64)
            print('Set new image for random shape: %s mu: %.3f sigma: %.3f' % (str(shran), mu, sigma))
            a = test_image(shape=shran, mu=mu, sigma=sigma)
            self.set_pixmap_from_arr(a, set_def=True) #, set_def=True, amin=None, amax=None, frmin=0.001, frmax=0.999)

        else:
            print(self.key_usage())


if __name__ == "__main__":
    import os
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1' #export LIBGL_ALWAYS_INDIRECT=1
    logging.basicConfig(format='[%(levelname).1s] %(filename)s L:%(lineno)03d %(message)s', level=logging.INFO)
    app = QApplication(sys.argv)
    w = TestGWViewImAx(signal_fast=False) # True)
    w.setGeometry(100, 50, 800, 800)
    w.setWindowTitle('ex_GWViewImAx')
    w.show()
    app.exec_()
    del w
    del app

# EOF
