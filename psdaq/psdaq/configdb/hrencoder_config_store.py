from psdaq.configdb.typed_json import cdict
import psdaq.configdb.configdb as cdb
import sys
import argparse
import IPython
import pyrogue as pr

def lookupValue(d,name):
    key = name.split('.',1)
    if key[0] in d:
        v = d[key[0]]
        if isinstance(v,dict):
            return lookupValue(v,key[1])
        elif isinstance(v,bool):
            return 1 if v else 0
        else:
            return v
    else:
        return None

class mcdict(cdict):
    def __init__(self, fn=None):
        super().__init__(self)

        self._yamld = {}
        if fn:
            print('Loading yaml...')
            self._yamld = pr.yamlToData(fName=fn)

    #  intercept the set call to replace value with yaml definition
    def init(self, prefix, name, value, type="INT32", override=False, append=False):
        v = lookupValue(self._yamld,name)
        if v:
            print('Replace {:}[{:}] with [{:}]'.format(name,value,v))
            value = v
        self.set(prefix+':RO.'+name+':RO', value, type, override, append)

def write_to_daq_config_db(args):
    create: bool = True
    dbname: str = 'configDB'

    db: str  = 'configdb' if args.prod else 'devconfigdb'
    url: str  = f'https://pswww.slac.stanford.edu/ws-auth/{db}/ws/'

    mycdb = cdb.configdb(url, args.inst, create,
                         root=dbname, user=args.user, password=args.password)
    mycdb.add_alias(args.alias)
    mycdb.add_device_config('hrencoder')

    top: mcdict = mcdict(args.yaml)
    top.setInfo('hrencoder', args.name, args.segm, args.id, 'No comment')
    top.setAlg('config', [0,1,0])

    help_str: str = (
        '-- user --\n'
        '  - delay_ns : Nanosecond delay to the encoder trigger signal. Adjust '
                       'for timing in the encoder.'
    )
    top.set('help:RO', help_str, 'CHARSTR')

    # Additional delay - cannot be negative if I understand correctly
    top.set('user.delay_ns', 105765, 'UINT32')

    top.set('expert.PauseThreshold', 16, 'UINT8')
    top.set('expert.TriggerDelay', 42, 'UINT32') # 185.7 MHz clocks

    mycdb.add_alias(args.alias)
    mycdb.modify_device(args.alias, top)


if __name__ == "__main__":
    args = cdb.createArgs().args
    write_to_daq_config_db(args)
