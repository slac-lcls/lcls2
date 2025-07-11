
"""Class :py:class:`UtilsFS` is a Image Viewer QWidget with control fields
==========================================================================

Usage ::

    from psana2.pyalgos.generic.UtilsFS import load_ndarray_from_file, list_of_files_in_dir, safe_listdir

Created on 2021-06-16 by Mikhail Dubrovin
"""

import logging
logger = logging.getLogger(__name__)

import os
import threading


def islink(path):
    return os.path.islink(path)


def isdir(path):
    return os.path.isdir(path)


def safe_listdir(directory, timeout_sec=5):
    contents = []
    t = threading.Thread(target=lambda: contents.extend(os.listdir(directory)))
    t.daemon = True  # don't delay program's exit
    t.start()
    t.join(timeout_sec)
    if t.is_alive():
        return None  # timeout
    return contents


def list_of_files_in_dir(dirname):
    return safe_listdir(dirname)


def list_of_files_in_dir_for_ext(dir, ext='.xtc2'):
    """Returns the list of files in the directory for specified extension or None if directory is None."""
    if dir is None: return []
    if not os.path.exists(dir): return []
    return sorted([f for f in safe_listdir(dir) if os.path.splitext(f)[1] == ext])


def list_of_pathes_in_dir_for_ext(dir, ext='.xtc2'):
    """Returns the list of pathes in the directory for specified extension or None if directory is None."""
    return [os.path.join(dir,f) for f in list_of_files_in_dir_for_ext(dir, ext)]


def list_of_files_in_dir_for_pattern(dir, pattern='-r0022'):
    """Returns the list of files in the directory for specified file name pattern or [] - empty list."""
    if dir is None: return []
    if not os.path.exists(dir): return []
    return sorted([os.path.join(dir,f) for f in safe_listdir(dir) if pattern in f])


def load_ndarray_from_file(path):
    ext = os.path.splitext(path)[1]
    if ext in ('.npy', ):
        import numpy as np
        return np.load(path)
    elif ext in ('.txt', '.data', '.dat'):
        from psana2.pscalib.calib.NDArrIO import load_txt#, save_txt
        return load_txt(path)
    else:
        logger.debug('not recognized file extension "%s", try to load as text file' % ext)
        from psana2.pscalib.calib.NDArrIO import load_txt
        return load_txt(path)


if __name__ == "__main__" :

    #from psana2.pyalgos.generic.NDArrUtils import info_ndarr

    import sys
    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d : %(message)s', level=logging.DEBUG)

    def test_list_of_instruments():
        sys.stdout.write('%s:\n  %s\n' % (sys._getframe().f_code.co_name, str(list_of_instruments())))

    USAGE = 'Use command: python %s <test-number>, where <test-number> = 0,1,2,...' % sys.argv[0]\
           + '\n  0: test_list_of_instruments'\
           + '\n  1: test_list_of_instruments'\
           + '\n'

    sys.stdout.write('\n%s\n' % USAGE)

    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    sys.stdout.write('%s Test %s %s\n' % (25*'_',tname, 25*'_'))
    if   tname == '0' : test_list_of_instruments()
    elif tname == '1' : test_list_of_instruments()
    else: ('Not-recognized test name: %s' % tname)
    sys.exit('End of test %s' % tname)

# EOF
