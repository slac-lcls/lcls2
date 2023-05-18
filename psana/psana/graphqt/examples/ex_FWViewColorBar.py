
from psana.graphqt.FWViewColorBar import *


class TestFWViewColorBar(FWViewColorBar):

    def test_new_color_table_reception(self):
        print('  FWViewColorBar.test_new_color_table_reception: %s' % str(self._ctab[:5]))

    def test_new_color_table_index_is_selected_reception(self, ind):
        print('  FWViewColorBar.test_new_color_table_index_is_selected_reception: %s' % str(self._ctab_ind))

    def key_usage(self):
        return 'Keys:'\
               '\n  ESC - exit'\
               '\n  R - reset color table 0'\
               '\n  N - set next color table'\
               '\n'

    def keyPressEvent(self, e):
        #print('keyPressEvent, key=', e.key())

        if not (self.change_mode & 1): return

        if   e.key() == Qt.Key_Escape:
            self.close()

        elif e.key() == Qt.Key_R:
            print('Reset original size')
            self.set_colorbar_table_ind(ctab_ind=0)

        elif e.key() == Qt.Key_N:
            print('Set next color table')
            self.set_colorbar_table_ind(ctab_ind=None)

        else:
            print(self.key_usage())


if __name__ == "__main__":

  import os
  import sys
  logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d : %(message)s', level=logging.DEBUG)

  os.environ['LIBGL_ALWAYS_INDIRECT'] = '1' #export LIBGL_ALWAYS_INDIRECT=1

  def test_fwviewcolorbar(tname):
    print('%s:' % sys._getframe().f_code.co_name)
    arr = np.random.random((1000, 100))
    #arr = image_with_random_peaks((1000, 1000))
    ctab = ct.color_table_rainbow(ncolors=1000, hang1=250, hang2=-20)
    #ctab = ct.color_table_monochr256()
    #ctab = ct.color_table_interpolated()

    app = QApplication(sys.argv)
    w = None
    if   tname == '0': w = TestFWViewColorBar(None, coltab=ctab, orient='H', change_mode=1, scale_ctl='H')
    elif tname == '1': w = TestFWViewColorBar(None, coltab=ctab, orient='V')
    else:
        print('test %s is not implemented' % tname)
        return

    w.setWindowTitle(w._name)

    w.connect_mouse_move_event(w.test_mouse_move_event_reception)
    w.connect_new_color_table_index_is_selected(w.test_new_color_table_index_is_selected_reception)
    w.connect_new_color_table(w.test_new_color_table_reception)

    w.show()
    app.exec_()

    del w
    del app


if __name__ == "__main__":
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print(50*'_', '\nTest %s' % tname)
    test_fwviewcolorbar(tname)
    sys.exit('End of Test %s' % tname)

# EOF
