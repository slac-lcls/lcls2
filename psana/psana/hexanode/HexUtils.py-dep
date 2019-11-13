#!@PYTHON@
#------------------------------
"""
Module :py:class:`HexUtils` a set of generic methods for hexanode project
=========================================================================

Usage ::

    import expmon.HexUtils as hu
    hu.print_kwargs(d) 
    d = hu.dict_from_str_srcchs(s)

See:
    - :class:`HexCalib`
    - :class:`HexDataIO`
    - :class:`HexDataIOExt`
    - :class:`HexDataPreProc`
    - :class:`HexUtils`

References:
    - `Index of expmon <https://lcls-psana.github.io/expmon/py-modindex.html>`_.
    - `Quad- and Hex-anode on confluence <https://confluence.slac.stanford.edu/display/PSDMInternal/Quad-+and+hex-+anode+detector+monitoring+software>`_.

Created on 2017-12-08 by Mikhail Dubrovin
"""
#------------------------------
# import os
# import sys
#------------------------------

def print_kwargs(d) :
    print '%s\n  kwargs:' % (40*'_')
    for k,v in d.iteritems() : print '  %10s : %10s' % (k,v)
    print 40*'_'

#------------------------------

def dict_from_str_srcchs(s) :
    """Converts string like
       "{'AmoETOF.0:Acqiris.0':(6,7,8,9,10,11),'AmoITOF.0:Acqiris.0':(0,)}"
       to the dictionary
    """
    #print 'srcchs:', s
    fields = s.lstrip('{').rstrip('}').split("),'")
    d = {}
    for f in fields :
       #print 'f:', f
       flds2 = f.split("':(")
       f0 = flds2[0].strip("'")
       f1 = flds2[1].lstrip('(').rstrip(')')
       #print 'f0: %s    f1: %s' % (f0, f1)
       d[f0] = [int(n) for n in f1.split(',') if n]
    return d

#------------------------------
#------------------------------
#------------------------------
#------------------------------

