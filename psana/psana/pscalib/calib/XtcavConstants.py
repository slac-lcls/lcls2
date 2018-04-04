import h5py
import numpy
import logging
"""
    Usage ::
    from psana.pscalib.calib.XtcavConstants import Load, Save
"""
class Empty(object):
    pass

class ConstantsStore(object):
    def __init__(self,obj,file):
        self.f = h5py.File(file,'w')
        self.cwd = ''
        for name in obj.__dict__:
            subobj = getattr(obj,name)
            self.dispatch(subobj,name)
        self.f.close()
    def pushdir(self,dir):
        '''move down a level and keep track of what hdf directory level we are in'''    
        
        self.cwd += '/'+dir
       
    def popdir(self):
        '''move up a level and keep track of what hdf directory level we are in'''
        self.cwd = self.cwd[:self.cwd.rfind('/')]
    def typeok(self,obj,name):
        '''check if we support serializing this type to hdf'''
        allowed = [dict,int,float,str,numpy.ndarray]
        return type(obj) in allowed
    def storevalue(self,v,name):
        '''persist one of the supported types to the hdf file'''
        self.f[self.cwd+'/'+name] = v
    def dict(self,d,name):
        '''called for every dictionary level to create a new hdf group name.
        it then looks into the dictionary to see if other groups need to
        be created'''
        if self.cwd is '':
            self.f.create_group(name)
        self.pushdir(name)
        for k in d.keys():
            self.dispatch(d[k],k)
        self.popdir()
    def dispatch(self,obj,name):
        '''either persist a supported object, or look into a dictionary
        to see what objects need to be persisted'''
        if type(obj) is dict:
            self.dict(obj,name)
        else:
            if self.typeok(obj,name):
                self.storevalue(obj,name)
            else:
                logging.warning('XTCAV Constants.py: variable "'+name+'" of type "'+type(obj).__name__+'" not supported')

class ConstantsLoad(object):
    def __init__(self,file):
        self.obj = Empty()
        self.f = h5py.File(file,'r')
        self.f.visititems(self.loadCallBack)
        self.f.close()
    def setval(self,name,obj):
        '''see if this hdfname has a / in it.  if so, create the dictionary
        object.  if not, set our attribute value.  call ourselves
        recursively to see if other dictionary levels exist.'''
        if '/' in name:
            dictname=name[:name.find('/')]
            remainder=name[name.find('/')+1:]

            if type(obj) is dict:
                indicator = dictname in obj
            else:
                indicator = hasattr(obj,dictname)

            if not indicator: 
                if type(obj) is dict:
                    obj[dictname]={}
                else:
                    setattr(obj,dictname,{})
               
            if type(obj) is dict:
                self.setval(remainder,obj[dictname])
            else:
                self.setval(remainder,getattr(obj,dictname))
        else:
            if type(obj) is dict:
                obj[name]=self.f[self.fullname].value
            else:
                setattr(obj,name,self.f[self.fullname].value)
    def loadCallBack(self,name,obj):
        '''called back by h5py routine visititems for each
        item (group/dataset) in the h5 file'''
        if isinstance(obj,h5py._hl.group.Group):
            return
        self.fullname = name
        self.setval(name,self.obj)

def Load(file):
    '''takes a string filename, and returns a constants object.'''
    c = ConstantsLoad(file)
    return c.obj

def Save(obj,file):
    '''store a constants object in an hdf5 file.  the object
    can be a hierarchy (defined by python dictionaries) and
    hdf5 supported types (int, float, numpy.ndarray, string).
    the hierarchy can be created by having one value of
    a dictionary itself be a dictionary.'''
    
    c = ConstantsStore(obj,file)

def dict_from_xtcav_calib_object(o):
    '''xtcav calibration object (a python object) has a list of attributes. 
    Each attribute may be simple or compaund type. 
    This method wraps all attributes except "__*" in the dictionary of pairs
    {<attr_name> : <attr_value>} and returns this dict.
    '''
    return dict([(name, getattr(o, name, None)) for name in dir(o) if name[:2] != '__'])


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


class ConstTest(object):
    def __init__(self):
        self.parameters= {
            'version' : 0,
            'darkreferencepath':'hello',
            'nb':12,
            'subdict':{'first' : 1, 'second' : 'two','three' : 'bahahah'}
            }
        
if __name__ == "__main__":
    ct = ConstTest()
    Save(ct,'ConstTest.h5')
    data = Load('ConstTest.h5')
    print('***',dir(data),data.parameters)
