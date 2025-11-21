import os
import sys
import argparse
import logging
import re
import socket
import subprocess
from p4p.client.thread import Context

#subnet = {'TMO':'10.0.0.4',
subnet = {'TMO':'daq-tmo-hsd-01',
          'RIX':'daq-rix-hsd-01',}

def main():
    parser = argparse.ArgumentParser(description='Make input selection for all HSDs in the hutch')
    parser.add_argument('-H', type=str, required=True, help='TMO,RIX,...', metavar='HUTCH')
    parser.add_argument('-I', type=int, required=False, help='Input selection (0=A0_2, 1=A1_3). Omit to see current setting', metavar='INPUT', default=None)
    parser.add_argument('-t', '--test',    action='store_true', help='test only.  No changes')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if not args.H.upper() in subnet:
        raise ValueError('Hutch selection unknown')

    if not (args.I is None or args.I in (0,1)):
        raise ValueError('Input selection not 0 or 1')

    #
    #  Search the known host for all HSD PVs
    #
    hsubnet  = socket.gethostbyname(subnet[args.H.upper()])
    os.environ['EPICS_PVA_ADDR_LIST'] = hsubnet
    os.environ['EPICS_PVA_AUTO_ADDR_LIST'] = 'NO'

    for k,v in os.environ.items():
        if 'EPICS_PVA' in k:
            logging.debug(f'{k}={v}')

    ctxt = Context('pva')
    pvs = []

    pvPrefix = f'DAQ:{args.H.upper()}:HSD'
    #  Find all the pv host processes
    result = subprocess.run(['pvlist'], capture_output=True)
    lines = result.stdout.decode('utf-8').split('\n')
    for line in lines:
        #  Filter on subnet
        ips = [ a.split(':')[0] for a in line.partition('tcp@[')[2].split(' ')[1:-1]]
        if not hsubnet in ips:
            continue

        guid = line.split(' ')[1]
        #  Find all the pvs served by this process
        result = subprocess.run(['pvlist',guid], capture_output=True)
        pvs = result.stdout.decode('utf-8').split('\n')
        for pv in pvs:
            #  Filter on an exact match
            if re.fullmatch(f'{pvPrefix}:1_[0-9,A-Z]*:[A,B]:RESET',pv):
                chn = pv.rsplit(':',1)[0]
                rst = ctxt.get(pv)
                input = rst['jesdsetup']
                if args.I is None:
                    logging.warning(f'Found {chn} input at {input}')
                else:
                    rst['jesdsetup'] = args.I
                    logging.debug(f'Put {rst}')
                    logging.warning(f'Setting {pv} input from {input} to {args.I}')
                    if not args.test:
                        ctxt.put(pv,rst,wait=False)

if __name__ == '__main__':
    main()
