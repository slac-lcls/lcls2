from psdaq.configdb.typed_json import cdict, copyValues
import psdaq.configdb.configdb as cdb
import pyrogue as pr
import numpy as np
import sys
import IPython
import argparse

elemRows = 144
elemCols = 192

if __name__ == "__main__":

    create = False
    dbname = 'configDB'     #this is the name of the database running on the server.  Only client care about this name.

    args = cdb.createArgs().args

    db = 'configdb' if args.prod else 'devconfigdb'
    mycdb = cdb.configdb(f'https://pswww.slac.stanford.edu/ws-auth/{db}/ws/', args.inst, create,
                         root=dbname, user=args.user, password=args.password)
    top = mycdb.get_configuration(args.alias, args.name+'_%d'%args.segm)

    if args.yaml is None:
        raise ValueError('--yaml flag required')

    (base,fn) = args.yaml.split(':')
    print(f'base {base}  fn {fn}')

    d = pr.yamlToData(fName=fn)
    print(f'keys {d}')

    copyValues(d[base],top,'expert')

    mycdb.modify_device(args.alias, top)
