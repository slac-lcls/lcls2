from psdaq.configdb.typed_json import cdict
import psdaq.configdb.configdb as cdb
from psdaq.configdb.get_config import update_config
import os
import io

def pyxpm_cdict():

    top = cdict()

    top.setAlg('config', [0,0,0])

    top.set('XTPG.CuDelay'   , 134682, 'UINT32')
    top.set('XTPG.CuBeamCode',     40, 'UINT8')
    top.set('XTPG.CuInput'   ,      1, 'UINT8')
    v = [10]*8
    top.set('PART.L0Delay', v, 'UINT32')

    return top

if __name__ == "__main__":
    create = True
    dbname = 'configDB'
    args = cdb.createArgs().args
    db   = 'configdb' if args.prod else 'devconfigdb'
    url  = f'https://pswww.slac.stanford.edu/ws-auth/{db}/ws/'

    mycdb = cdb.configdb(url, args.inst, create,
                         root=dbname, user=args.user, password=args.password)

    top = pyxpm_cdict()
    top.setInfo('xpm', args.name, None, 'serial1234', 'No comment')
    
    if args.update:
        cfg = mycdb.get_configuration(args.alias, args.name)
        cfg['XTPG']['CuInput'] = 1
        top = update_config(cfg, top.typed_json(), args.verbose)

    if not args.dryrun:
        if create:
            mycdb.add_alias(args.alias)
            mycdb.add_device_config('xpm')
        mycdb.modify_device(args.alias, top)

