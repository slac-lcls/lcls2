from psdaq.cas.config_scan_base import ConfigScanBase
from psdaq.configdb.get_config import cdb
import json
import os
import sys

def listParams(d,name):
    result = []
    name = '' if name is None else name+'.'
    for k,v in d.items():
        if 'PacketRegisters' in k:
            pass
        elif 'Hr10kTAsic' in k:
            if k=='Hr10kTAsic0':
                result.extend(listParams(v,f'{name}Hr10kTAsic'))
            else:
                pass
        elif isinstance(v,dict):
            result.extend(listParams(v,f'{name}{k}'))
        else:
            result.extend([f'{name}{k}'])
    return result
            
def main():

    aargs = [('-P',{'default':None,'help':'parameter to scan (omit to get full list'}),
             ('--linear',{'type':float,'nargs':3,'help':'linear scan over range [0]:[1] in steps of [2]'})]
    scan = ConfigScanBase(aargs)
             
    args = scan.args

    #  Validate scan parameter
    #  Lookup the configuration in the database
    prod  = True
    if args.hutch not in ('tmo','rix','asc'):
        hutch = 'tst'
    else:
        hutch = args.hutch
             
    del sys.argv[1:]
    dbargs = cdb.createArgs().args
    dbargs.inst = hutch
    dbargs.prod = False
    dbargs.name = 'epixhr'
    dbargs.segm = 0
    dbargs.user = os.environ['USER']
    create = False
    db = 'configdb' # if dbargs.prod else 'devconfigdb'
    mycdb = cdb.configdb(f'https://pswww.slac.stanford.edu/ws-auth/{db}/ws/', dbargs.inst, create,
                         root='configDB', user=dbargs.user, password=dbargs.password)
    top = mycdb.get_configuration(dbargs.alias, dbargs.name+'_%d'%dbargs.segm)

    d = top['expert']['EpixHR']
    l = listParams(d,None)

    keys = None
    if args.P is not None and args.P in l:
        if 'Hr10kTAsic' in args.P:
            keys = [f'{args.detname}:expert.EpixHR.Hr10kTAsic{i}.{args.P[11:]}' for i in range(4)]
        elif 'PacketRegisters' in args.P:
            pass
        else:
            keys = [f'{args.detname}:expert.EpixHR.{args.P}']
        if keys is None:
            print('Invalid parameter {args.P}')

    usage = keys is None

    if usage:
        print('Valid parameters are:')
        for i in l:
            print(i)
        return

    if args.linear:
        print(f'linear: {args.linear}')
        def steps():
            metad = {'detname':args.detname, 'scantype':args.scantype}
            d = {}
            for value in np.arange(*args.linear):
                for k in keys:
                    d[k] = int(value)
                yield (d, value, json.dumps(metad))

    else:
        raise RuntimeError('Must specify scan type (--linear,)')

    scan.run(keys,steps)

if __name__ == '__main__':
    main()
