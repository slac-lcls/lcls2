"""
    Usage ::
    from psana.pscalib.calib.XtcavUtils import dict_from_xtcav_calib_object,\
                                               xtcav_calib_object_from_dict, Empty
"""

import logging
logger = logging.getLogger(__name__)

import numpy as np
from psana.pscalib.calib.XtcavConstants import *

import base64

#--------------------

def numpy_scalar_types():
    """np.sctypes = 
    {'int': [numpy.int8, numpy.int16, numpy.int32, numpy.int64],
     'uint': [numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64],
     'float': [numpy.float16, numpy.float32, numpy.float64, numpy.float128],
     'complex': [numpy.complex64, numpy.complex128, numpy.complex256],
     'others': [bool, object, bytes, str, numpy.void]}
    np.sctypeDict ??
    """
    return tuple(np.sctypes['int'] + np.sctypes['uint'] + np.sctypes['float'] + np.sctypes['complex'])

NUMPY_SCALAR_TYPES = numpy_scalar_types()

#--------------------

def dict_from_xtcav_calib_object(o):
    '''xtcav calibration object (a python object) has a list of attributes. 
    Each attribute may be simple or compaund type. 
    This method wraps all attributes except "__*" in the dictionary of pairs
    {<attr_name> : <attr_value>} and returns this dict.
    '''
    return dict([(name, getattr(o, name, None)) for name in dir(o) if name[:2] != '__'])

#--------------------

def xtcav_calib_object_from_dict(d):
    '''Converts input dictionary in xtcav calibration object and returns this object.
    Top level of dict (k,v) pairs is converted in the empty object attributes using setattr(o,str(k),v)    
    '''
    o = Empty()
    #assert isinstance(d, dict), 'Input parameter is not a python dict: %s' % str(d)
    if not isinstance(d, dict) :
        raise TypeError('Input parameter is not a python dict: %s' % str(d))
        #logging.warning('Input parameter is not a python dict: %s' % str(d))
        #return o
    for k,v in d.items() :
        setattr(o,str(k),v)
    return o

#------------------------------

def load_xtcav_calib_file(fname) :
    """Returns dict made of xtcav object loaded from hdf5 file by XtcavConstants.Load method.
    """
    logger.info('Load xtcav calib object from file: %s'%fname)
    o = Load(fname)
    return dict_from_xtcav_calib_object(o)

#--------------------

def compare_dicts(d1, d2, gap='  '):
    print('%sCompare two dictionaries:'%gap)
    allowed = [dict, int, float, str, bytes, numpy.ndarray, numpy.int64, numpy.float64]
    for k1,v1 in d1.items() :
        s_type = '"%s"' % str(type(v1)).split("'")[1]
        s_key = '%skey: %s values of type %s' % (gap, k1.ljust(20), s_type.ljust(16))
        if not (type(v1) in allowed) :
            logging.warning('%s of type %s are not compared' % (s_key, str(type(v1))))
        v2 = d2.get(k1, None)
        if isinstance(v1, dict) : 
            print(s_key)
            compare_dicts(v1,v2,gap='%s  '%gap)
        elif isinstance(v1, numpy.ndarray) : print('%s are equal: %s' % (s_key, numpy.array_equal(v1, v2)))
        else : print('%s are equal: %s' % (s_key, v1 == v2))

#--------------------

def jasonify_numpy(nda):
    """ Returns dict of numpy adday data and metadata.
    """
    return {'type' :'nd',
            'shape':str(nda.shape),
            'size' :str(nda.size),
            'dtype':nda.dtype.str,
            'data' :str(nda.tobytes()) #.replace("'", '"')
           }
    #        'data' :str(nda.tobytes().replace("'", '"')
    #        'data' :(nda.tobytes()).decode('ascii') # 'utf-8')
    #        'data' :''.join(chr(b) for b in nda.tobytes())

#--------------------

def jasonify_dict(d, offset='  '):
    """ Returns dict of strings for k, v, saves data and data types for scalars.
    """
    logger.debug('%sXtcavUtils.jasonify_dict:' % offset)
    for k,v in d.items() :
        logger.debug('%sk:%s  type(v):%s' % (offset, k.ljust(16), type(v)))
        if   isinstance(v,dict) : jasonify_dict(v, offset = offset+'  ')
        elif isinstance(v, np.ndarray) : d[k] = jasonify_numpy(v)
        #elif isinstance(v, bytes)      : d[k] = str(v)
        #elif isinstance(v, bytes)      : d[k] = ''.join(chr(b) for b in v)
        elif isinstance(v, bytes)      : d[k] = str(v) #.replace("'", '"')
        elif isinstance(v, NUMPY_SCALAR_TYPES) : d[k] = {'type' :'sc', 'dtype':v.dtype.str, 'data':str(v)} 
        elif not isinstance(v, str) : d[k] = str(v)

#--------------------

def info_dict(d, offset='  ', s=''):
    """ returns (str) dict content
    """
    s = '%s\n%sinfo_dict' % (s, offset)
    for k,v in d.items() :
        if isinstance(v,dict) : s = info_dict(v, offset = offset+'  ', s=s)
        s = '%s\n%sk:%s t:%s v:%s' % (s, offset, str(k).ljust(10), type(v), str(v)[:120])
    return s

#--------------------

def print_dict(d, offset='  '):
    """ prints dict content
    """
    print('%sprint_dict' % offset)
    for k,v in d.items() :
        if isinstance(v,dict) : print_dict(v, offset = offset+'  ')
        print('%sk:%s t:%s v:%s' % (offset, str(k).ljust(10), type(v), str(v)[:120]))

#--------------------
#--------------------
#--------------------
#--------------------

if __name__ == "__main__":
  def test_const():
    ct = ConstTest()
    Save(ct,'ConstTest.h5')
    data = Load('ConstTest.h5')
    print('***',dir(data),data.parameters)

#--------------------

  def test_jasonify_numpy():
    from psana.pyalgos.generic.NDArrGenerators import np, aranged_array, random_standard
    from psana.pyalgos.generic.NDArrUtils import print_ndarr # info_ndarr
    #nda = random_standard(shape=(4,6), mu=100, sigma=10, dtype=np.float)
    nda = aranged_array(shape=(3,4), dtype=np.float) # uint32)
    print_ndarr(nda, 'nda', first=0, last=12)

    d = jasonify_numpy(nda)
    print('jasonify_numpy: %s' % d)

#--------------------

  def test_jasonify_dict():
      #fname = '/reg/d/psdm/AMO/amox23616/calib/'\
      #        'Xtcav::CalibV1/XrayTransportDiagnostic.0:Opal1000.0/lasingoffreference/56-end.data'
      #fname = '/reg/g/psdm/detector/data_test/calib/'\
      #        'Xtcav::CalibV1/XrayTransportDiagnostic.0:Opal1000.0/lasingoffreference/56-end.data'
      fname = '/reg/g/psdm/detector/data_test/calib/'\
              'Xtcav::CalibV1/XrayTransportDiagnostic.0:Opal1000.0/pedestals/104-end.data'
      o = Load(fname)
      dico = dict_from_xtcav_calib_object(o)
      jasonify_dict(dico)
      #print('dico: %s' % dico)
      print(80*'_')
      print_dict(dico)
      print(80*'_')

#--------------------
        
if __name__ == "__main__":

    import sys
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    tname = sys.argv[1] if len(sys.argv) > 1 else '1'
    logger.info('%s\nTest %s:' % (50*'_',tname))
    if   tname == '0' : test_const()
    elif tname == '1' : test_jasonify_numpy()
    elif tname == '2' : test_jasonify_dict()
    else : logger.info('Not-recognized test name: %s' % tname)
    sys.exit('End of test %s' % tname)

#--------------------
