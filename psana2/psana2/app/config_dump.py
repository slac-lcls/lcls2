from psana2 import DataSource
from psana2 import container
from psana2.psexp.utils import DataSourceFromString
import numpy as np
import argparse

def dump(obj, attrlist):
    allattrs = dir(obj)
    usefulattrs=[attr for attr in allattrs if (not attr.startswith('_') and attr != 'help')]
    for attr in usefulattrs:
        val = getattr(obj, attr)
        attrlist.append(attr)
        if type(val) in [int, float, np.ndarray, str]:
            print('.'.join(attrlist)+':', val)
        elif type(val) is container.Container:
            dump(val, attrlist)
        attrlist.pop(-1)

def config_dump():

    parser = argparse.ArgumentParser(description='LCLS2 Configuration Dump Utility')
    parser.add_argument("dsname", help="psana datasource experiment/run (e.g. exp=xppd7114,run=43) or xtc2 filename or shmem=<my_shmem_identifier>")
    parser.add_argument("detname", help="Detector name selected from output of 'detnames' command")
    parser.add_argument("datatype", help="Data type selected from output of 'detnames' command")
    parser.add_argument("-s","--segments", action='append', help="segment to dump from 'detnames -i' command. use multiple '-s' options to dump multiple segments", type=int, default=[])
    args=parser.parse_args()
    ds = DataSourceFromString(args.dsname)
    myrun = next(ds.runs())
    det = myrun.Detector(args.detname)
    cfgs = getattr(det,args.datatype)._seg_configs()

    attrlist=[]
    if len(args.segments)>0:
        for seg in args.segments:
            dump(cfgs[seg], attrlist)
    else:
        for myobj in cfgs.values():
            dump(myobj, attrlist)
