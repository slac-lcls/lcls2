import sys
import os
from multiprocessing import Pool
import subprocess
import argparse

#
#  Scan-specific data
#
hsd_nodes = [('daq-rix-hsd-01','1a'),
             ('daq-rix-hsd-01','1b'),
             ('daq-tmo-hsd-01','01'),
             ('daq-tmo-hsd-01','1a'),
             ('daq-tmo-hsd-01','1b'),
             ('daq-tmo-hsd-01','3d'),
             ('daq-tmo-hsd-01','3e'),
             ('daq-tmo-hsd-01','88'),
             ('daq-tmo-hsd-01','89'),
             ('daq-tmo-hsd-01','b1'),
             ('daq-tmo-hsd-01','b2'),
             ('daq-tmo-hsd-01','da'),
             ('daq-tmo-hsd-02','41'),]
tdet_nodes = ['drp-srcf-cmp001',
              'drp-srcf-cmp002',
              'drp-srcf-cmp003',
              'drp-srcf-cmp010',
              'drp-srcf-cmp025',
              'drp-srcf-cmp028',
              'drp-srcf-cmp029',
              'drp-srcf-cmp030',
              'drp-srcf-cmp032',
              'drp-srcf-cmp037',
              'drp-srcf-cmp042',
              'drp-srcf-cmp043',]
xpm_nodes = [('drp-srcf-mon001','10.0.1.102'),
             ('drp-srcf-mon001','10.0.1.104'),
             ('drp-srcf-mon001','10.0.1.105'),
             ('drp-srcf-mon001','10.0.1.107'),
             ('daq-tmo-hsd-01' ,'10.0.3.103'),
             ('daq-tmo-hsd-01' ,'10.0.3.105'),
             ('daq-rix-hsd-01' ,'10.0.2.103'),
             ('daq-rix-hsd-01' ,'10.0.2.102'),
             ('drp-neh-ctl002' ,'10.0.5.102'),
             ('drp-neh-ctl002' ,'10.0.5.104'),]
drp_hsd_nodes = [('drp-srcf-cmp005','0'),
                 ('drp-srcf-cmp005','1'),
                 ('drp-srcf-cmp017','0'),
                 ('drp-srcf-cmp017','1'),
                 ('drp-srcf-cmp018','0'),
                 ('drp-srcf-cmp018','1'),
                 ('drp-srcf-cmp019','0'),
                 ('drp-srcf-cmp019','1'),
                 ('drp-srcf-cmp020','0'),
                 ('drp-srcf-cmp020','1'),
                 ('drp-srcf-cmp021','0'),
                 ('drp-srcf-cmp021','1'),
                 ('drp-srcf-cmp022','0'),
                 ('drp-srcf-cmp022','1'),
                 ('drp-srcf-cmp023','0'), # pepex
                 ('drp-srcf-cmp023','1'), # pepex
                 ('drp-srcf-cmp024','0'),
                 ('drp-srcf-cmp024','1'),
                 ('drp-srcf-cmp046','0'),
                 ('drp-srcf-cmp046','1'),
                 ('drp-srcf-cmp048','0'),
                 ('drp-srcf-cmp048','1'),
                 ('drp-srcf-cmp050','0'),
                 ('drp-srcf-cmp050','1'),]

def parse1(fname):
    words = fname.split('.')
    host = words[0].split('/')[-1]
    return (host,)

def parse2(fname):
    words = fname.split('.')
    host = words[0].split('/')[-1]
    dev  = words[1].split('_')[1]
    return (host,dev)

def parse3(fname):
    words = fname.split('.')
    host = words[0].split('/')[-1]
    dev  = words[1].split('_')[1]
    lane = words[2]
    return (host,dev,lane)

def parsex(fname):
    ip = '.'.join(fname.split('.')[1:5])
    chn = fname.split('.')[5]
    return (ip,chn)

on = None
dn = None

scans = {'hsd-tim':{'nodes': hsd_nodes,
                    'command':lambda node : f'ssh {node[0]} {dn}/launch-python.sh hsd.py --bathtub --dev /dev/datadev_{node[1]} --write {on} --link=-1',
                    'field_names':('HOST','DEV'),
                    'field_parse':parse2},
         'hsd-pgp':{'nodes': hsd_nodes,
                    'command':lambda node : f'ssh {node[0]} {dn}/launch-python.sh hsd.py --bathtub --dev /dev/datadev_{node[1]} --write {on}',
                    'field_names':('HOST','DEV','LANE'),
                    'field_parse':parse3},
         'drp-tdet':{'nodes': tdet_nodes,
                     'command':lambda node : f'ssh {node} {dn}/launch-python.sh tdet-drp.py --bathtub --write {on}',
                     'field_names':('HOST',),
                     'field_parse':parse1},
         'drp-hsd':{'nodes': drp_hsd_nodes,
                    'command':lambda node : f'ssh {node[0]} {dn}/launch-python.sh hsd-drp.py --dev /dev/datadev_{node[1]} --bathtub --write {on}',
                    'field_names':('HOST','DEV','LANE'),
                    'field_parse':parse3},
         'xpm':{'nodes': xpm_nodes,
                'command':lambda node : f'ssh {node[0]} {dn}/launch-python.sh xpm.py --ip {node[1]} --bathtub --write {on}',
                'field_names':('IP','LINK'),
                'field_parse':parsex},}

#
#  Common infrastructure
#
parser = argparse.ArgumentParser(prog=sys.argv[0], description='Collect GTH bathtub scans for many nodes')
parser.add_argument('--scan', default='tdet', help=f'Scan type {scans.keys()}')
parser.add_argument('--output', default='.', help='Output file directory', metavar='OPATH')
parser.add_argument('--readonly', action='store_true', help='Just read the directory')
parser.add_argument('--limit', type=float, default=1.e-15, help='BER alarm limit')
args = parser.parse_args()

nodes   = scans[args.scan]['nodes']
command = scans[args.scan]['command']

dn = os.path.dirname(os.path.realpath(__file__))

if not os.path.isdir(args.output):
    result=None
    while(result!='Y' and result!='N'):
        result=input(f'{args.output} is not a directory.  Create it (Y/N)?')
    if result=='Y':
        os.mkdir(args.output)
    else:
        print(f'Aborting.')
        exit(1)

on = f'{os.path.dirname(os.path.realpath(args.output))}/{args.output}'

def launch(node):
    cmd=f'xterm -T {".".join(node)} -e {command(node)}'
    print(cmd)
    subprocess.call(cmd, shell=True)

if not args.readonly:
    with Pool(processes=None) as pool:
        result = pool.map_async(launch, nodes)
        result.wait()

#
#  Make a pretty table
#
from prettytable import PrettyTable
import glob

# Tabulate the results
table = PrettyTable()
table.field_names = scans[args.scan]['field_names']+('BER',)
fnames = glob.glob(f'{on}/*.dat')
fnames.sort()
data = []
lber = []
for fname in fnames:
    fields = scans[args.scan]['field_parse'](fname)
    f = open(fname,'r')
    ber = float(f.read().split(':')[1])
    lber.append(ber)
    sber = f'\033[0;33;40m{ber:.2g}\033[0m' if ber > args.limit else f'{ber:.2g}'
    data.append(fields+(sber,))

for i,row in enumerate(data):
    table.add_row(row,divider = (i+1)<len(data) and row[0]!=data[i+1][0])

print(table)

#
# Histogram the BER
#
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(tight_layout=True)
ax.hist(np.log10(lber), bins=40)
plt.xlabel('Log10 BER')
plt.savefig(f'{on}/all.png')
plt.show()
