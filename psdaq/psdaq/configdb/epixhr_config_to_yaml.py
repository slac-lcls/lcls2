from psdaq.configdb.typed_json import cdict
import psdaq.configdb.configdb as cdb
import pyrogue as pr
import numpy as np
import sys
import IPython
import argparse

def intToBool(d,types,key):
    if isinstance(d[key],dict):
        for k,value in d[key].items():
            intToBool(d[key],types[key],k)
    elif types[key]=='boolEnum':
        d[key] = False if d[key]==0 else True

def dictToYaml(d,types,keys,dev,path,name):
    v = {}
    for key in keys:
        v[key] = d[key]
        intToBool(v,types,key)

    nd = {'ePixHr10kT':{'EpixHR':v}}
    yaml = pr.dataToYaml(nd)
    fn = path+name+'.yml'
    f = open(fn,'w')
    f.write(yaml)
    f.close()
    setattr(dev,'filename'+name,fn)

class EpixHR(object):
    def __init__(self):
        pass

class test(object):
    def __init__(self):
        self.EpixHR = EpixHR()

if __name__ == "__main__":

    create = False
    dbname = 'configDB'     #this is the name of the database running on the server.  Only client care about this name.

    args = cdb.createArgs().args

    db = 'configdb' if args.prod else 'devconfigdb'
    mycdb = cdb.configdb(f'https://pswww.slac.stanford.edu/ws-auth/{db}/ws/', args.inst, create,
                         root=dbname, user=args.user, password=args.password)
    top = mycdb.get_configuration(args.alias, args.name+'_%d'%args.segm)

    epixHR      = top           ['expert']['EpixHR']
    epixHRTypes = top[':types:']['expert']['EpixHR']
    path = '/tmp/epixhr'
    cbase = test()

    dictToYaml(epixHR,epixHRTypes,['MMCMRegisters'  ],cbase.EpixHR,path,'MMCM')
    dictToYaml(epixHR,epixHRTypes,['PowerSupply'    ],cbase.EpixHR,path,'PowerSupply')
    dictToYaml(epixHR,epixHRTypes,['RegisterControl'],cbase.EpixHR,path,'RegisterControl')
    for i in range(4):
        dictToYaml(epixHR,epixHRTypes,['Hr10kTAsic{}'.format(i)],cbase.EpixHR,path,'ASIC{}'.format(i))
    dictToYaml(epixHR,epixHRTypes,['PacketRegisters{}'.format(i) for i in range(4)],cbase.EpixHR,path,'PacketReg')
    dictToYaml(epixHR,epixHRTypes,['TriggerRegisters'],cbase.EpixHR,path,'TriggerReg')

    print(vars(cbase.EpixHR))
