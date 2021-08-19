from psana import DataSource
from psana import container
from psana.psexp.utils import DataSourceFromString
import numpy as np
import argparse

def dump(obj, attrlist):
    allattrs = dir(obj)
    usefulattrs=[attr for attr in allattrs if not attr.startswith('_')]
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
    parser.add_argument("dsname", help="psana datasource experiment/run (e.g. exp=xppd7114,run=43) or xtc2 filename or shmem='my_shmem_identifier'")
    parser.add_argument("detname", help="Detector name selected from output of 'detnames' command")
    parser.add_argument("datatype", help="Data type selected from output of 'detnames' command")
    parser.add_argument("-s","--segments", nargs='*', help="space-separated list of segment numbers from 'detnames -i' command (e.g. 4 6 7)", type=int, default=[])
    args=parser.parse_args()
    ds = DataSourceFromString(args.dsname)
    myrun = next(ds.runs())
    det = myrun.Detector(args.det)
    cfgs = getattr(det,args.datatype)._seg_configs()

    attrlist=[]
    if len(args.segments)>0:
        for seg in args.segments:
            dump(cfgs[seg], attrlist)
    else:
        for myobj in cfgs.values():
            dump(myobj, attrlist)
