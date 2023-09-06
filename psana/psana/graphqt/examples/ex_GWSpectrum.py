#!/usr/bin/env python

from psana.graphqt.GWSpectrum import *

class TestGWSpectrum(GWSpectrum):

    KEY_USAGE = 'Keys:'\
             '\n  ESC - exit'\
             '\n  R - reset original size'\
             '\n  N - set new spectrum'\
             '\n'

    def __init__(self, **kwa):
        GWSpectrum.__init__(self, **kwa)
        print(self.KEY_USAGE)

    def keyPressEvent(self, e):
      logger.debug('==  keyPressEvent key=%s' % e.key())
      if   e.key() == Qt.Key_Escape:
          print('Close app')
          self.close()

      elif e.key() == Qt.Key_R:
          r = self.whi.scene_rect()
          print('Reset original size whi.scene_rect: %s' % qu.info_rect_xywh(r))
          self.on_but_reset()

      elif e.key() == Qt.Key_N:
          shran = np.maximum(random_standard(shape=(2,), mu=500, sigma=500, dtype=int), (100,100))
          mu, sigma = random_standard(shape=(2,), mu=500, sigma=200, dtype=np.float64)
          print('\nset new histogram for img shape: %s mu: %.3f sigma: %.3f' % (str(shran), mu, sigma))
          a = test_image(shape=shran, mu=mu, sigma=sigma)
          self.set_spectrum_from_arr(a)

      else:
          print(self.KEY_USAGE)


if __name__ == "__main__":
    import os
    os.environ['LIBGL_ALWAYS_INDIRECT'] = '1' #export LIBGL_ALWAYS_INDIRECT=1

    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(name)s : %(message)s', level=logging.INFO)

    app = QApplication(sys.argv)
    w = TestGWSpectrum(signal_fast=True)
    w.setGeometry(100, 50, 300, 800)
    w.setWindowTitle('ex_GWSpectrum')

    w.connect_histogram_scene_rect_changed(w.test_histogram_scene_rect_changed)
    w.wcbar.connect_new_color_table(w.wcbar.test_new_color_table_reception)

    w.show()
    app.exec_()

    del w
    del app

# EOF
