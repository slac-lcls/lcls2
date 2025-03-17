from psdaq.configdb.typed_json import cdict
import psdaq.configdb.configdb as cdb
from psdaq.configdb.get_config import get_config_with_params
import psdaq.pyxpm.autosave as autosave
import os
import io

if __name__ == "__main__":
    create = True
    dbname = 'configDB'
    args = cdb.createArgs().args
    db   = 'configdb' if args.prod else 'devconfigdb'
    url  = f'https://pswww.slac.stanford.edu/ws-auth/{db}/ws/'
    print(f'url {url}')
    
    mycdb = cdb.configdb(url, args.inst, create,
                         root=dbname, user=args.user, password=args.password)
    try:
        cfg = mycdb.get_configuration(args.alias, args.name)
        print(f'cfg {cfg}')
    except:
        print(f'Creating alias {args.alias}')
        mycdb.add_alias(args.alias)
        mycdb.add_device_config('xpm')
        cfg = {'PART':{'L0Delay':[0,0,0,0,0,0,0,0,]},
               'XTPG':{'CuBeamCode':140,'CuDelay':0,'CuInput':1}}
        print(f'created cfg {cfg}')

    autosave.set(args.name,f'{url},{dbname},{args.inst},{args.alias}',None)
    for i,v in enumerate(cfg['PART']['L0Delay']):
        autosave.add(f'{args.name}:PART:{i}:L0Delay',v)
    for k,v in cfg['XTPG'].items():
        autosave.add(f'{args.name}:XTPG:{k}',v)
    autosave.dump()
    autosave.save()

