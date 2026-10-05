from p4p.client.thread import Context
from psdaq.configdb.tsdef import *

class PatternStats(object):

    def __init__(self, pv, group):
        self.ctx = Context('pva')
        self.base = f'{pv}:PART:{group}'
        self.patt = f'{pv}:PATT:GROUPS'
        self.group = group

        #  Setup the pattern monitor for FR 1H
        self.ctx.put(f'{pv}:PATT:L0Select',0)
        self.ctx.put(f'{pv}:PATT:L0Select_FixedRate',FixedIntvsDict['1H']['marker'])
        
    def _setPV(self,name,val):
        rval = self.ctx.put(f'{self.base}:{name}',val)
        
    def setup(self,code=None,destNames=None):
        if code is None:
            self._setPV('L0Select',0)
            self._setPV('L0Select_FixedRate',FixedIntvs.index(1))
        else:
            self._setPV('L0Select',2)
            self._setPV('L0Select_EventCode',code)
            
        if destNames is None:
            self._setPV('DstSelect',1)
        else:
            self._setPV('DstSelect',0)
            self._setPV('DstSelect_Mask',destnValue(destNames))

    def get(self):
        rval = self.ctx.get(self.patt)
        d = {}
        for a in ('Sum','First','Last','MinIntv','MaxIntv'):
            d[a] = getattr(rval.value,a)[self.group]
        return d
            
if __name__=='__main__':
    import argparse
    import sys
    import time
    
    parser = argparse.ArgumentParser(prog=sys.argv[0], description='test pattern fetch')
    parser.add_argument('-P', default='DAQ:NEH:XPM:5', help='XPM PV')
    parser.add_argument('--group', default=7, type=int, help='readout group')
    parser.add_argument('--code', default=None, type=int, help='event code')
    parser.add_argument('--beam', default=None, nargs='+', help='include destinations comma-separated list of {BSYD,HXR,SXR}')
    args = parser.parse_args()
    print(f'args {args}')
    
    stats = PatternStats(args.P,args.group)
    stats.setup(args.code,args.beam)
    time.sleep(3)
    print(stats.get())
