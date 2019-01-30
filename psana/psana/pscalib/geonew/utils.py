
"""
Module :py:class:`utils` contains re-usable utilities
=====================================================

Usage::
    from psana.pscalib.geonew.utils object_from_python_code, load_exec_python_code, load_textfile, memorize

    txt = load_textfile(fname)
    dict_local_vars = load_exec_python_code(fname)
    o = object_from_python_code(fname, objname)

    # @memorize() # single object caching decorator
    # def pixel_sizes_slow(self) : ...

This software was developed for the LCLS2 project.
If you use all or part of it, please give an appropriate acknowledgment.

Created on 2019-01-30 by Mikhail Dubrovin
"""

#----------

import logging
logger = logging.getLogger(__name__)

#----------

def load_textfile(path) :
    """Returns (str) text from file.
    """
    f=open(path, 'r')
    recs = f.read() # f.readlines()
    f.close() 
    return recs

#----------

def load_exec_python_code(fname) :
    """ Loads and exec python code from input file.
        Returns dlocal - dictionary of local variables. 
        Acess ex. dlocal['seg_cspad2x1'], or dlocal['source']
    """
    source = load_textfile(fname)
    #logger.debug('source:\n%s' % source)
    dglobal = None 
    dlocal = locals()
    exec(source, dglobal, dlocal)
    return dlocal

#----------

def object_from_python_code(fname, objname) :
    """Returns python object defined in the specified file with code.
       Source code (str) and object name are preserved with object itself
    """
    d = load_exec_python_code(fname)
    o = d.get(objname, None)
    o._source = d.get('source', None)
    o._objname = objname
    return o

#----------

def memorize(dic={}) :
    """Caching decorator
       E.g.: @memorize(); ...
    """
    def deco_memorize(f) :
        def wrapper(*args, **kwargs):
            fid = id(f)
            v = dic.get(fid, None)
            if v is None :
                v = dic[fid] = f(*args, **kwargs)
            return v
        return wrapper
    return deco_memorize

#----------
