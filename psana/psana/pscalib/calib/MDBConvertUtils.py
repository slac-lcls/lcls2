"""
    Usage ::
    from psana.pscalib.calib.MDBConvertUtils import numpy_scalar_types, compare_dicts,\
                                                    serialize_numpy, serialize_dict, info_dict, print_dict
"""

import logging
logger = logging.getLogger(__name__)

import numpy as np

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

def serialize_numpy(nda):
    """ Returns dict of numpy array data and metadata.
    """
    return {'type' :'nd',
            'shape':str(nda.shape),
            'dtype':nda.dtype.str,
            'data' :str(nda.tobytes()) #.replace("'", '"')
           }

#--------------------

def is_none(v,msg):
    if v is None : 
        logger.debug('deserialize_numpy: paremeter "%s" is None' % msg)
        return True

#--------------------

def deserialize_numpy(d):
    """ Returns numpy array from serialized in dict numpy array.
    """
    t = d.get('type', None)
    if t != 'nd' :
        logger.debug('deserialize_numpy: wrong type "%s"' % str(t))
        return None

    data = d.get('data', None)
    if is_none(data, 'data'):
        return None 

    dtype = d.get('dtype', None)
    if is_none(dtype, 'dtype'):
        return None

    shape = d.get('shape', None)
    if is_none(shape, 'shape'):
        return None

    nda = np.fromstring(data, dtype=dtype)
    nda.shape = eval(shape)
    return nda

#--------------------

def serialize_dict(d, offset='  '):
    """ Returns dict of strings for k, v, saves data and data types for scalars.
    """
    logger.debug('%sserialize_dict:' % offset)
    for k,v in d.items() :
        logger.debug('%sk:%s  type(v):%s' % (offset, k.ljust(16), type(v)))
        if   isinstance(v,dict) : serialize_dict(v, offset = offset+'  ')
        elif isinstance(v, np.ndarray) : d[k] = serialize_numpy(v)
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

#--------------------
  from psana.pyalgos.generic.NDArrGenerators import np, aranged_array, random_standard

  def test_serialize_numpy():
    from psana.pyalgos.generic.NDArrUtils import print_ndarr # info_ndarr
    #nda = random_standard(shape=(4,6), mu=100, sigma=10, dtype=np.float)
    nda = aranged_array(shape=(2,3), dtype=np.float) # uint32)
    print_ndarr(nda, 'nda', first=0, last=12)
    d = serialize_numpy(nda)
    print('serialize_numpy: %s' % d)

#--------------------

  def test_serialize_dict():
      dico = {'val':123,
              'nda1': aranged_array(shape=(3,4), dtype=np.int),
              'nda2': random_standard(shape=(2,3), mu=100, sigma=10, dtype=np.float)
             }
      serialize_dict(dico)
      #print('dico: %s' % dico)
      print(80*'_')
      print('serialize_dict:')
      print_dict(dico)
      print(80*'_')

#--------------------
        
if __name__ == "__main__":

    import sys
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    logger.info('%s\nTest %s:' % (50*'_',tname))
    if   tname == '0' : test_serialize_numpy()
    elif tname == '1' : test_serialize_dict()
    else : logger.info('Not-recognized test name: %s' % tname)
    sys.exit('End of test %s' % tname)

#--------------------
