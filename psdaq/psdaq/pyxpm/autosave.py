import psdaq.configdb.configdb as cdb
from psdaq.configdb.typed_json import cdict
from psdaq.configdb.get_config import get_config_with_params
from p4p.client.thread import Context
import logging

autosave = None
countRst = 3600

def add(name, value):
    autosave.add(name,value)

def modify(name, value):
    autosave.modify(name,value)

def save():
    autosave.save()

def restore():
    autosave.restore()

def update():
    autosave.update()

def dump():
    autosave.dump()

class Autosave(object):
    def __init__(self, dev, db, lock):
        self.dev     = dev
        self.lock    = lock
        self.countdn = 0
        self.dict    = {}
        self.ctxt    = Context('pva', nt=None)
        if db:
            self.db  = db.split(',',4) #(db_url, db_name, db_instrument, db_alias)
        else:
            self.db  = None

    def dump(self):
        for k,v in self.dict.items():
            print(f'dict[{k}] = {v}')

    # add PV name, value pairs to go into configdb
    def add(self, name, value):
        if self.dev:
            if self.dev==name[:len(self.dev)]:
                rem = '.'.join(name[len(self.dev)+1:].split(':'))
                self.dict[rem] = value
                self.countdn = countRst
            else:
                logging.warning(f'Autosave.add {name} not a child of {self.dev}')
                
    def modify(self, name, value):
        if self.dev==name[:len(self.dev)]:
            rem = '.'.join(name[len(self.dev)+1:].split(':'))
            if rem in self.dict:
                self.dict[rem] = value
                self.countdn = countRst
        
    def _cdict(self):
        top = cdict()
        top.setInfo('xpm', self.dev, None, 'serial1234', 'No comment')
        top.setAlg('config', [0,0,0])

        for k,v in self.dict.items():
            top.set('PV.'+k, v, 'UINT32')
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
            for k,v in self.dict.items():
                print(f'  {k}: {v}')

    def _restore_dict(self,d,name):
        for k,v in d.items():
            if isinstance(v,dict):
                self._restore_dict(v,f'{name}:{k}')
            # list only allowed at the lowest level
            elif isinstance(v,list) and isinstance(v[-1],dict):
                for i,w in enumerate(v):
                    self._restore_dict(v[i],f'{name}:{k}:{i}')
            else:
                n = f'{name}:{k}'
                print(f'restoring {n} {v}')
                self.ctxt.put(n,v)

    #  retrieve PV name, value pairs and post them
    def restore(self):
        if self.db:
            db_url, db_name, db_instrument, db_alias = self.db
            logging.info('db {:}'.format(self.db))
            logging.info('url {:}  name {:}  instr {:}  alias {:}'.format(db_url,db_name,db_instrument,db_alias))
            logging.info('device {:}'.format(self.dev))
            init = get_config_with_params(db_url, db_instrument, db_name, db_alias, self.dev)
            logging.info('cfg {:}'.format(init))
            # exchange . for : to recreate PVs?
            if 'PV' in init:
                self._restore_dict(init['PV'],self.dev)
            
    def update(self):
        if self.countdn > 0:
            self.countdn -= 1
            if self.countdn == 0 and self.db:
                self.save()
                #d = self._cdict().typed_json()
                #print(f'update {d}')
                #self._restore_dict(d['PV'],self.dev)

def set(name,db,lock):
    global autosave
    autosave = Autosave(name,db,lock)
