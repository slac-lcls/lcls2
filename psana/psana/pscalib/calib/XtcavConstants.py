
# Copy of
# https://github.com/lcls-psana/xtcav/blob/master/src/Constants.py

import h5py
import numpy
import logging

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
                obj[name]=self.f[self.fullname][()] #.value
            else:
                setattr(obj,name,self.f[self.fullname][()]) #.value)
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

class ConstTest(object):
    def __init__(self):
        self.parameters= {
            'version' : 0,
            'darkreferencepath':'hello',
            'nb':12,
            'subdict':{'first' : 1, 'second' : 'two','three' : 'bahahah'}
            }

#--------------------

if __name__ == "__main__":
    ct = ConstTest()
    Save(ct,'ConstTest.h5')
    data = Load('ConstTest.h5')
    print('***',dir(data),data.parameters)

#--------------------
