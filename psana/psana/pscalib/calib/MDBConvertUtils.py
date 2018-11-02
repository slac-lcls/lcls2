"""
    Usage ::
    from psana.pscalib.calib.MDBConvertUtils import numpy_scalar_types, compare_dicts, is_none,\
                                                    info_dict, print_dict,\
                                                    serialize_dict, deserialize_dict,\
                                                    serialize_numpy_array, deserialize_numpy_array,\
                                                    serialize_numpy_value, serialize_value, deserialize_value

    # Main methods of this module for I/O parameter (dict) d:
    serialize_dict(d)   # converts all dict values to str or bytes
    deserialize_dict(d) # doing conversion opposite to serialize_dict
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
TYPES_OF_INT = ('int', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64')
TYPES_OF_FLOAT = ('float', 'float16', 'float32', 'float64','float128')

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

def serialize_value(v) :
    return {'type' :'sc', 'dtype':type(v).__name__, 'data':str(v)} 

#--------------------

def serialize_numpy_value(v) :
    return {'type' :'sc', 'dtype':str(v.dtype), 'data':str(v)} 

#--------------------

def deserialize_value(d) :
    """ Operation opposite to serialize_numpy_value(v)
    """
    t = d.get('type', None)
    if t != 'sc' :
        logger.debug('deserialize_value: wrong type "%s", expected "sc"' % str(t))
        return None

    dtype = d.get('dtype', None)
    if is_none(dtype, 'dtype') : return None

    data = d.get('data', None)
    if is_none(data, 'data') : return None

    return int(data)   if dtype in TYPES_OF_INT   else\
           float(data) if dtype in TYPES_OF_FLOAT else\
           str(data)   if dtype == 'str' else\
           bool(data)  if dtype == 'bool' else\
           eval(data)

#--------------------

def is_none(v,msg):
    if v is None : 
        logger.debug('deserialize_numpy_array/value: paremeter "%s" is None' % msg)
        return True

#--------------------

def serialize_numpy_array(nda):
    """ Returns dict for numpy array data and metadata.
        nda.dtype is like (str) 'uint32'
    """
    return {'type' : 'nd',
            'shape': str(nda.shape),
            'size' : str(nda.size), 
            'dtype': str(nda.dtype), 
            'data' : nda.tobytes() # (bytes)
           }
#            'data' : nda.tobytes().decode('utf-16') # converts to str

#--------------------

def deserialize_numpy_array(d):
    """ Returns numpy array from serialized in dict numpy array.
    """
    t = d.get('type', None)
    if t != 'nd' :
        logger.debug('deserialize_numpy_array: wrong type "%s", expected "nd"' % str(t))
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

    size = d.get('size', None)
    if is_none(size, 'size') : return None
    size = int(size)

    nda = np.frombuffer(data, dtype=dtype, count=size) # .encode('utf-16','ignore')
    #nda = np.fromstring(data.encode('utf-16'), dtype=dtype, count=size) # 'utf-16','ignore'
    # bytearray(data.encode('utf-16')
    nda.shape = eval(shape)
    return nda

#--------------------

def serialize_dict(d):
    """ Converts i/o dict values to str.
    """
    #logger.debug('serialize_dict:')
    for k,v in d.items() :
        #logger.debug('k:%s  type(v):%s' % (k.ljust(16), type(v)))
        if   isinstance(v, str)                 : continue
        elif isinstance(v, dict)                : serialize_dict(v)
        elif isinstance(v, np.ndarray)          : d[k] = serialize_numpy_array(v)
        elif isinstance(v, bytes)               : d[k] = str(v)
        elif isinstance(v, (int,float,str,bool)): d[k] = serialize_value(v)
        elif isinstance(v, NUMPY_SCALAR_TYPES)  : d[k] = serialize_numpy_value(v)
        elif not isinstance(v, str)             : d[k] = str(v)

#--------------------

def deserialize_dict(d):
    """ Returns deserialized dict, operation opposite to serialize_dict method.
    """
    if not isinstance(d, dict) : 
        logger.warning('deserialize_dict: input value type "%s" IS NOT a dict' % type(d))
        return

    for k,v in d.items() :
        #logger.debug('k:%s  type(v):%s' % (k.ljust(16), type(v)))
        if isinstance(v, dict) : 
            type = v.get('type', None)
            if   type == 'nd' :
                d[k] = deserialize_numpy_array(v)
            elif type == 'sc' :
                d[k] = deserialize_value(v)
            else :
                deserialize_dict(v)

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
  from psana.pyalgos.generic.NDArrGenerators import aranged_array, random_standard
  from psana.pyalgos.generic.NDArrUtils import print_ndarr # info_ndarr

  def test_serialize_numpy_array():
    nda = random_standard(shape=(4,6), mu=100, sigma=10, dtype=np.float32)
    #nda = aranged_array(shape=(2,3), dtype=np.float) # uint32)
    print_ndarr(nda, 'nda', first=0, last=12)
    d = serialize_numpy_array(nda)
    print('serialize_numpy_array: %s' % d)
    nda2 = deserialize_numpy_array(d)
    print_ndarr(nda, 'de-serialized nda', first=0, last=12)

#--------------------

  def test_serialize_dict():
      d = {'val':123,
           'nda1': aranged_array(shape=(3,4), dtype=np.int),
           'nda2': random_standard(shape=(2,3), mu=100, sigma=10, dtype=np.float)
          }
      print('initial dict:')
      print_dict(d)
      serialize_dict(d)
      print(80*'_')
      print('serialized dict:')
      print_dict(d)
      print(80*'_')
      deserialize_dict(d)
      print('deserialized dict:')
      print_dict(d)
      print(80*'_')

#--------------------

  def test_serialize_numpy_value():
      nda = aranged_array(shape=(2,3), dtype=np.uint32)
      v = nda[1,1]
      print('value:', v, ' v.dtype:', v.dtype, ' v.dtype.str:', v.dtype.str)
      d = serialize_numpy_value(v)
      print('serialize_numpy_value:', d)
      v2 = deserialize_value(d)
      print('deserialize_value:', v2)

#--------------------

  def test_serialize_value():
      v = float(6)
      print('value:', v, ' type(v).__name__:', type(v).__name__)
      d = serialize_value(v)
      print('serialize_value:', d)
      v2 = deserialize_value(d)
      print('deserialize_value:', v2)

#--------------------

  def usage() : 
      return 'Use command: python %s <test-number>, where <test-number> = 0,1,2,3' % sys.argv[0]\
           + '\n  0: test_serialize_numpy_value'\
           + '\n  1: test_serialize_numpy_array'\
           + '\n  2: test_serialize_dict'\
           + '\n  3: test_serialize_value'

#--------------------
        
if __name__ == "__main__":

    import sys
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    logger.info('\n%s\n' % usage())
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    logger.info('%s\nTest %s:' % (50*'_',tname))
    if   tname == '0' : test_serialize_numpy_value()
    elif tname == '1' : test_serialize_numpy_array()
    elif tname == '2' : test_serialize_dict()
    elif tname == '3' : test_serialize_value()
    else : logger.info('Not-recognized test name: %s' % tname)
    sys.exit('End of test %s' % tname)

#--------------------
