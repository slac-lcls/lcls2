#------------------------------
""" Set of utilities involving os, sys, etc
Usage ::
    import expmon.EMUtils as emu
    srcs = emu.list_of_files_in_dir(dirname)

@version $Id: EMUtils.py 13157 2017-02-18 00:05:34Z dubrovin@SLAC.STANFORD.EDU $

@author Mikhail S. Dubrovin
"""
#------------------------------
import sys
import os
from expmon.PSNameManager import nm
#------------------------------

def list_of_experiments(direxp=None) :
    """ Returns list of experiments in experimental directiory defined through configuration parameters.

    Parameters
    direxp : str - directory of experiments for particular instrument  # e.g. '/reg/d/psdm/XPP'
    """
    #ptrn = cp.instr_name.value().lower() if pattern is None else pattern # e.g. 'xpp'
    dir  = nm.dir_exp() if direxp is None else direxp  # e.g. '/reg/d/psdm/XPP'
    ptrn = dir.rstrip('/').rsplit('/',1)[1].lower()    # e.g. 'xpp'
    #print 'dir: %s  ptrn: %s' % (dir, ptrn)
    ldir = sorted(os.listdir(dir))
    #print 'XXX list_of_experiments:', ldir
    return [e for e in ldir if e[:3] == ptrn]

#------------------------------

def list_of_files_in_dir(dirname) :
    return os.listdir(dirname)

#------------------------------

def list_of_files_in_dir_for_ext(dir, ext='.xtc') :
    """Returns the list of files in the directory for specified extension or None if directory is None."""
    if dir is None : return []
    if not os.path.exists(dir) : return [] 
    return sorted([f for f in os.listdir(dir) if os.path.splitext(f)[1] == ext])
    
#------------------------------

def list_of_pathes_in_dir_for_ext(dir, ext='.xtc') :
    """Returns the list of pathes in the directory for specified extension or None if directory is None."""
    return [os.path.join(dir,f) for f in list_of_files_in_dir_for_ext(dir, ext)]
    
#------------------------------

def list_of_files_in_dir_for_pattern(dir, pattern='-r0022') :
    """Returns the list of files in the directory for specified file name pattern or [] - empty list."""
    if dir is None : return []
    if not os.path.exists(dir) : return []
    return sorted([os.path.join(dir,f) for f in os.listdir(dir) if pattern in f])

#------------------------------

def list_of_int_from_list_of_str(list_in) :
    """Converts  ['0001', '0202', '0203', '0204',...] to [1, 202, 203, 204,...]
    """
    return [int(item.lstrip('0')) for item in list_in]

#------------------------------

def list_of_str_from_list_of_int(list_in, fmt='%04d') :
    """Converts [1, 202, 203, 204,...] to ['0001', '0202', '0203', '0204',...]
    """
    return [fmt % item for item in list_in]

#------------------------------

def list_of_runs_in_xtc_dir(dirxtc=None) :
    dir = nm.dir_xtc() if dirxtc is None else dirxtc  # e.g. '/reg/d/psdm/XPP/xpptut13/xpp'
    xtcfiles = list_of_files_in_dir_for_ext(dir, ext='.xtc')
    runs = [f.split('-')[1].lstrip('r') for f in xtcfiles]
    return set(runs)

#------------------------------
#------------------------------
#------------------------------

def src_type_alias_from_cfg_key(key) :
    """Returns striped object 'EventKey(type=None, src='DetInfo(CxiDs2.0:Cspad.0)', alias='DsdCsPad')'"""
    return k.src(), k.type(), k.alias()

#------------------------------
#------------------------------
#------------------------------

def test_list_of_files_in_dir() :
    print '%s:' % sys._getframe().f_code.co_name
    lfiles = list_of_files_in_dir('/reg/d/psdm/sxr/sxrtut13/xtc/')
    for fname in lfiles : print fname

#------------------------------

def test_list_of_files_in_dir_for_pattern() :
    print '%s:' % sys._getframe().f_code.co_name
    lfiles = list_of_files_in_dir_for_pattern('/reg/d/psdm/cxi/cxitut13/xtc/', pattern='-r0011')
    for fname in lfiles : print fname

#------------------------------

def test_list_of_files_in_dir_for_ext() :
    print '%s:' % sys._getframe().f_code.co_name
    lfiles = list_of_files_in_dir_for_ext('/reg/d/psdm/sxr/sxrtut13/xtc/', ext='.xtc')
    for fname in lfiles : print fname

#------------------------------

def test_list_of_str_from_list_of_int() :
    print '%s:' % sys._getframe().f_code.co_name
    print list_of_str_from_list_of_int([1, 202, 203, 204], fmt='%04d')

#------------------------------

def test_list_of_int_from_list_of_str() :
    print '%s:' % sys._getframe().f_code.co_name
    print list_of_int_from_list_of_str(['0001', '0202', '0203', '0204'])

#------------------------------

def test_list_of_experiments(tname) :
    print '%s:' % sys._getframe().f_code.co_name
    lexps = []
    if   tname == '0': lexps = list_of_experiments() # uses config parameters
    elif tname == '1': lexps = list_of_experiments('/reg/d/psdm/XPP')
    else : return

    print 'list_of_experiments():'
    for i,e in enumerate(lexps) :
        print e,
        if not (i+1)%10 : print ''

#------------------------------

def test_list_of_runs_in_xtc_dir() :
    print '%s:' % sys._getframe().f_code.co_name
    print list_of_runs_in_xtc_dir()

#------------------------------

def test_all(tname) :

    from expmon.EMConfigParameters import cp
    nm.set_config_pars(cp)

    lexps = []
    if tname == '0': test_list_of_experiments(tname)
    if tname == '1': test_list_of_experiments(tname) 
    if tname == '2': test_list_of_str_from_list_of_int() 
    if tname == '3': test_list_of_int_from_list_of_str()
    if tname == '4': test_list_of_files_in_dir()
    if tname == '5': test_list_of_files_in_dir_for_ext()
    if tname == '6': test_list_of_files_in_dir_for_pattern()
    if tname == '7': test_list_of_runs_in_xtc_dir()
    else : return

#------------------------------

if __name__ == "__main__" :
    import sys; global sys
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print 50*'_', '\nTest %s' % tname
    test_all(tname)
    sys.exit('End of Test %s' % tname)

#------------------------------
