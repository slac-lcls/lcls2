#!/usr/bin/env python

import sys
import pyrogue as pr
import argparse

import logging
import time
from datetime import datetime
import socket

from p4p.server import Server, StaticProvider
from p4p.server.thread import SharedPV
from p4p.nt import NTScalar

def tdet(args):
    from psdaq.utils import enable_l2si_drp
    import l2si_drp

    root = l2si_drp.DrpTDetRoot(pollEn=False, 
                                devname=args.dev) #, qsa=args.qsa)
    # Start the system
    root.start()

    axiv = root.PcieControl.DevKcu1500.AxiPcieCore.AxiVersion
    axiv.printStatus()
    fwver = axiv.FpgaVersion.get()

    return (root.PcieControl.DevKcu1500.TDetTiming.TimingFrameRx,
            root.PcieControl.DevKcu1500.TDetTiming.TriggerEventManager.XpmMessageAligner,
            getattr(root.PcieControl.DevKcu1500.TDetTiming.TriggerEventManager,'TriggerEventBuffer[0]'))

def lppa(args):
    from psdaq.utils import enable_lcls2_pgp_pcie_apps
    import lcls2_pgp_pcie_apps
    import lcls2_pgp_fw_lib.shared as shared

    root = lcls2_pgp_pcie_apps.DevRoot(dev = args.dev,
                                       enLclsI = False,
                                       enLclsII = True,
                                       yamlFileLclsI = None,
                                       yamlFileLclsII = None,
                                       startupMode = None,
                                       standAloneMode = False,
                                       pgp4  = args.pgp4,
                                       dataVc = None,
                                       pollEn = False,
                                       initRead = False,
                                       pcieBoardType = args.boardType)

    # Start the system
    shared.Root.start(root)

    axiv = root.DevPcie.AxiPcieCore.AxiVersion
    axiv.printStatus()
    fwver = axiv.FpgaVersion.get()

    return (root.DevPcie.Hsio.TimingRx.TimingFrameRx,
            root.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner,
            getattr(root.DevPcie.Hsio.TimingRx.TriggerEventManager,'TriggerEventBuffer[0]'))

def lepx(args):
    from psdaq.utils import enable_lcls2_epix_hr_pcie
    import lcls2_epix_hr_pcie
    import lcls2_pgp_fw_lib.shared as shared

    root = lcls2_epix_hr_pcie.DevRoot(dev = args.dev,
                                      enLclsI = False,
                                      enLclsII = True,
                                      yamlFileLclsI = None,
                                      yamlFileLclsII = None,
                                      startupMode = None,
                                      standAloneMode = False,
                                      pgp4  = args.pgp4,
                                      pollEn = False,
                                      initRead = False,
                                      pcieBoardType = args.boardType,
                                      serverPort = 0)

    # Start the system
    shared.Root.start(root)

    axiv = root.DevPcie.AxiPcieCore.AxiVersion
    axiv.printStatus()
    fwver = axiv.FpgaVersion.get()

    return (root.DevPcie.Hsio.TimingRx.TimingFrameRx,
            root.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner,
            getattr(root.DevPcie.Hsio.TimingRx.TriggerEventManager,'TriggerEventBuffer[0]'))

def ludp(args):
    from psdaq.utils import enable_lcls2_udp_pcie_apps
    import lcls2_udp_pcie_apps
    import lcls2_pgp_fw_lib.shared as shared

    root = lcls2_udp_pcie_apps.DevRoot(dev = args.dev,
                                       defaultFile = None,
                                       standAloneMode = False,
                                       pollEn = False,
                                       initRead = False,
                                       pcieBoardType = args.boardType)

    # Start the system
    shared.Root.start(root)

    axiv = root.DevPcie.AxiPcieCore.AxiVersion
    axiv.printStatus()
    fwver = axiv.FpgaVersion.get()

    return (root.DevPcie.Hsio.TimingRx.TimingFrameRx,
            root.DevPcie.Hsio.TimingRx.TriggerEventManager.XpmMessageAligner,
            getattr(root.DevPcie.Hsio.TimingRx.TriggerEventManager,'TriggerEventBuffer[0]'))

class MyProvider(StaticProvider):
    def __init__(self, name):
        super(MyProvider,self).__init__(name)
        self.pvdict = {}

    def add(self,name,pv):
        self.pvdict[name] = pv
        super(MyProvider,self).add(name,pv)
        logging.info(f'Added PV {name}')

class DefaultPVHandler(object):

    def __init__(self, ctype='UINT32'):
        self._ctype   = ctype

    def put(self, pv, op):
        postedval = op.value()
        logging.debug('DefaultPVHandler.put ',pv,postedval['value'])
        postedval['timeStamp.secondsPastEpoch'], postedval['timeStamp.nanoseconds'] = divmod(float(time.time_ns()), 1.0e9)
        pv.post(postedval)
        op.done()

def updatePv(pv,v,timev):
    if v is not None:
        value = pv.current()
        if value['value']!=v:
            value['value'] = v
            value['timeStamp.secondsPastEpoch'], value['timeStamp.nanoseconds'] = timev
            pv.post(value)

argBool = lambda s: s.lower() in ['true', 't', 'yes', '1']

def main():
    global pvdb
    pvdb = {}     # start with empty dictionary
    global prefix
    prefix = ''

    parser = argparse.ArgumentParser(prog=sys.argv[0], description='host PVs for TDET')

    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')
    parser.add_argument('--type',default='DrpTDet', help='app type (DrpTDet,Lcls2Pgp,Lcls2Epix)')
    parser.add_argument('--dev', default='/dev/datadev_1', help='device file')
    parser.add_argument('--pgp4', type=argBool, default=0, help='true = PGPv4, false = PGP2b')
    parser.add_argument('--boardType', default='XilinxKcu1500', help='(None, SlacPgpCardG4, XilinxKcu1500, XilinxVariumC1100)')
#    parser.add_argument('--qsa', action='store_true', help='T=SFP,F=QSFP')
    parser.add_argument('--period', type=float, default=1.0, help='Update period')

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)s:%(name)s:%(message)s')

    hname = socket.gethostname().upper().replace('-',':')+':'

    provider = MyProvider(__name__)

    def addPV(name,ctype,init=0):
        handler = DefaultPVHandler()
        pv = SharedPV(initial=NTScalar(ctype).wrap(init), handler=handler)
        provider.add(hname+name, pv)
        return pv

    trx = None
    xma = None
    teb = None
    if args.type == 'DrpTDet':
        trx, xma, teb = tdet(args)
    if args.type == 'Lcls2Pgp':
        trx, xma, teb = lppa(args)
    if args.type == 'Lcls2Epix':
        trx, xma, teb = lepx(args)
    if args.type == 'Lcls2Udp':
        trx, xma, teb = ludp(args)
    if xma is None:
        raise ValueError(f'Type {args.type} unknown')

    pvs = []
#    pvs.append((addPV('FIDS','I'),trx.FidCount))
#    pvs.append((addPV('LINK','I'),trx.RxLinkUp))
    pvs.append((addPV('TXID','I'),xma.TxId))
    pvs.append((addPV('RXID','I'),xma.RxId))
    pvs.append((addPV('PINGID','I'),teb.LastPingId if hasattr(teb,'LastPingId') else teb.LastPartitionAddr))

    # process PVA transactions
    with Server(providers=[provider]):
        try:
            while True:
                prev  = time.perf_counter()

                timev = divmod(float(time.time_ns()), 1.0e9)
                for pv in pvs:
                    updatePv(pv[0], pv[1].get(), timev)

                curr  = time.perf_counter()
                delta = prev+args.period-curr
                if delta>0:
                    time.sleep(delta)
        except KeyboardInterrupt:
            pass

if __name__ == '__main__':
    main()
