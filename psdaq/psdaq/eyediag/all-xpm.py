import sys
import os
from multiprocessing import Pool
import subprocess
import argparse
import glob

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Collect GTH bathtub scans for all XPM channels')
parser.add_argument('--output', default='.', help='Output file directory', metavar='OPATH')
parser.add_argument('--readonly', action='store_true', help='Just read the directory')
parser.add_argument('--limit', type=float, default=1.e-15, help='BER limit')
args = parser.parse_args()

dn = os.path.dirname(os.path.realpath(__file__))

if not os.path.isdir(args.output):
    result=None
    while(result!='Y' and result!='N'):
        result=input(f'{args.output} is not a directoy.  Create it (Y/N)?')
    if result=='Y':
        os.mkdir(args.output)
    else:
        print(f'Aborting.')
        exit(1)

on = f'{os.path.dirname(os.path.realpath(args.output))}/{args.output}'

def launch(node):
    cmd=f'xterm -T {node[0]}:{node[1]} -e ssh {node[0]} {dn}/all-xpm.sh {node[1]} {on}'
#    cmd=f'ssh {node[0]} {dn}/all-xpm.sh {node[1]} {on}'
    print(cmd)
    subprocess.call(cmd, shell=True)

nodes = [('drp-srcf-mon001','10.0.1.102'),
         ('drp-srcf-mon001','10.0.1.104'),
         ('drp-srcf-mon001','10.0.1.105'),
         ('drp-srcf-mon001','10.0.1.107'),
         ('daq-tmo-hsd-01' ,'10.0.3.103'),
         ('daq-tmo-hsd-01' ,'10.0.3.105'),
         ('daq-rix-hsd-01' ,'10.0.2.103'),
         ('daq-rix-hsd-01' ,'10.0.2.102'),
         ('drp-neh-ctl002' ,'10.0.5.102'),
         ('drp-neh-ctl002' ,'10.0.5.104'),]

if not args.readonly:
    with Pool(processes=None) as pool:
        result = pool.map_async(launch, nodes)
        result.wait()

from prettytable import PrettyTable

# Tabulate the results
table = PrettyTable()
table.field_names = ['IP','CHN','BER']
fnames = glob.glob(f'{on}/*.dat')
fnames.sort()
data = []
lber = []
for fname in fnames:
    ip = '.'.join(fname.split('.')[1:5])
    chn = fname.split('.')[5]
    f = open(fname,'r')
    ber = float(f.read().split(':')[1])
    lber.append(ber)
    sber = f'\033[0;33;40m{ber:.2g}\033[0m' if ber > args.limit else f'{ber:.2g}'
    data.append([ip,chn,sber])

for i,row in enumerate(data):
    table.add_row(row,divider = (i+1)<len(data) and row[0]!=data[i+1][0])

print(table)

# Histogram the BER
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(tight_layout=True)
ax.hist(np.log10(lber), bins=40)
plt.xlabel('Log10 BER')
plt.savefig(f'{on}/all-xpm.png')
plt.show()
