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
            
def main():
    import argparse
    import sys
    import time
    
    parser = argparse.ArgumentParser(prog=sys.argv[0], description='fetch pattern statistics of an event code or beam')
    parser.add_argument('-P', default='DAQ:NEH:XPM:5', help='XPM PV name (default DAQ:NEH:XPM:5)')
    parser.add_argument('--group', default=7, type=int, help='readout group (default 7)')
    parser.add_argument('--code', default=None, type=int, help='event code')
    parser.add_argument('--beam', default=None, nargs='+', help='include destinations comma-separated list of {BSYD,HXR,SXR} [only with SC Timing input]')
    args = parser.parse_args()
    print(f'args {args}')

    masters = ('DAQ:NEH:XPM:2','DAQ:NEH:XPM:3','DAQ:FEH:XPM:4','DAQ:FEH:XPM:2')
    if args.P in masters:
        raise ValueError(f'Using any of {masters} may interfere with operations')
    
    stats = PatternStats(args.P,args.group)
    stats.setup(args.code,args.beam)
    time.sleep(3)
    print(stats.get())

if __name__=='__main__':
    main()
    
