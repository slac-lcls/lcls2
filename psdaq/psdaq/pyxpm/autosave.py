import psdaq.configdb.configdb as cdb
from psdaq.configdb.typed_json import cdict
from psdaq.configdb.get_config import get_config_with_params
from p4p.client.thread import Context
import logging
import pprint

countRst = 3600
#countRst = 60  # testing

class Autosave(object):
    def __init__(self, dev, db, lock, norestore):
        self.lock    = lock
        self.countdn = 0
        #  PV save/restore
        self.dev     = dev
        self.pvdict  = {}
        self.ctxt    = Context('pva', nt=None)
        #  Object save/restore
        self.objdict = {}
        self._norestore = norestore

        if db:
            self.db  = db.split(',',4) #(db_url, db_name, db_instrument, db_alias)
        else:
            self.db  = None

    def dump(self):
        for k,v in self.pvdict.items():
            print(f'pvdict[{k}] = {v}')

    # add PV name, value pairs to go into configdb
    def add(self, name, value, ctype='UINT32'):
        #print(f'autosave.add {name} {value} {ctype}')
        if self.dev:
            if self.dev==name[:len(self.dev)]:
                rem = '.'.join(name[len(self.dev)+1:].split(':'))
                self.pvdict[rem] = (value, ctype)
                self.countdn = countRst
            else:
                logging.warning(f'Autosave.add {name} not a child of {self.dev}')
                
    # add object, json pairs to go into configdb
    def addJson(self, name, obj, value):
        #print(f'autosave.addJson {name} {obj} {value}')
        self.objdict[name] = (obj,value)
        self.countdn = countRst

    def modify(self, name, value):
        if self.dev==name[:len(self.dev)]:
            rem = '.'.join(name[len(self.dev)+1:].split(':'))
            if rem in self.pvdict:
                self.pvdict[rem] = (value, self.pvdict[rem][1])
                self.countdn = countRst
        
    def _cdict(self):
        top = cdict()
        top.setInfo('xpm', self.dev, None, 'serial1234', 'No comment')
        top.setAlg('config', [0,0,0])

        for k,v in self.pvdict.items():
            value = v[0]
            #  cdict doesn't support list of strings
            if isinstance(v[0],list) and v[1]=='CHARSTR':
                value = '|'.join(v[0])
            top.set('PV.'+k, value, v[1])

        for k,v in self.objdict.items():
            top.set('OBJ.'+k, v[1], 'CHARSTR')

        return top

    # save config
    def save(self):
        if self.db:
            logging.info('Updating {}'.format(self.db))
            db_url, db_name, db_instrument, db_alias = self.db
            mycdb = cdb.configdb(db_url, db_instrument, True, db_name, user=db_instrument+'opr')
            mycdb.add_device_config('xpm')

            top = self._cdict()

            if not db_alias in mycdb.get_aliases():
                mycdb.add_alias(db_alias)

            try:
                mycdb.modify_device(db_alias, top)
            except:
                pass
                
        else:
            print('--Autosave (no db)--')
            for k,v in self.pvdict.items():
                print(f'  {k}: {v}')
            for k,v in self.objdict.items():
                print(f'  {k}: {v[1]}')

    def _restore_pvdict(self,d,name):
        for k,v in d.items():
            if isinstance(v,dict):
                self._restore_pvdict(v,f'{name}:{k}')
            # list only allowed at the lowest level
            elif isinstance(v,list) and isinstance(v[-1],dict):
                for i,w in enumerate(v):
                    self._restore_pvdict(v[i],f'{name}:{k}:{i}')
            else:
                n = f'{name}:{k}'
                #  cdict doesn't support list of strings
                if isinstance(v,str) and '|' in v:
                    value = v.split('|')
                else:
                    value = v
                self.ctxt.put(n,value)

    def _restore_objdict(self,d):
        for k,v in d.items():
            if k in self.objdict:
                self.objdict[k][0].restore(v)
            else:
                print(f'skipping restore {k}')

    #  retrieve PV name, value pairs and post them
    def restore(self):
        if self.db and not self._norestore:
            db_url, db_name, db_instrument, db_alias = self.db
            logging.info('db {:}'.format(self.db))
            logging.info('url {:}  name {:}  instr {:}  alias {:}'.format(db_url,db_name,db_instrument,db_alias))
            logging.info('device {:}'.format(self.dev))
            init = get_config_with_params(db_url, db_instrument, db_name, db_alias, self.dev)
            s = pprint.pformat(init)
            logging.info('cfg {:}'.format(s))
            # exchange . for : to recreate PVs?

            if 'PV' in init:
                self._restore_pvdict(init['PV'],self.dev)
            else:
                logging.info(f'No PV in init {init}')

            if 'OBJ' in init:
                self._restore_objdict(init['OBJ'])
            else:
                logging.info(f'No OBJ in init {init}')
            
    def update(self):
        if self.countdn > 0:
            self.countdn -= 1
            if self.countdn == 0:
                self.save()

class NoAutosave(object):
    def __init__(self):
        pass

    def dump(self):
        pass

    # add PV name, value pairs to go into configdb
    def add(self, name, value, ctype='UINT32'):
        pass

    def addJson(self, name, obj, value):
        print(f'objdict[{name}] = ({obj},{value})')

    def modify(self, name, value):
        pass
        
    def save(self):
        pass

    def restore(self):
        pass
            
    def update(self):
        pass

ottosave = NoAutosave()

def add(name, value, ctype='UINT32'):
    ottosave.add(name,value,ctype)

def addJson(name, obj, value):
    ottosave.addJson(name,obj,value)

def modify(name, value):
    ottosave.modify(name,value)

def save():
    ottosave.save()

def restore():
    ottosave.restore()

def update():
    ottosave.update()

def dump():
    ottosave.dump()

def set(name,db,lock,norestore=False):
    global ottosave
    ottosave = Autosave(name,db,lock,norestore)
