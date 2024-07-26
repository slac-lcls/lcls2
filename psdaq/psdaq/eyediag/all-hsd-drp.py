import sys
import os
from multiprocessing import Pool
import subprocess
import argparse

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Collect GTH bathtub scans for all TDET channels')
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
    cmd=f'xterm -T {node} -e ssh {node} {dn}/all-hsd-drp.sh {on}'
    print(cmd)
    subprocess.call(cmd, shell=True)

nodes = ['drp-srcf-cmp005',
         'drp-srcf-cmp017',
         'drp-srcf-cmp018',
         'drp-srcf-cmp019',
         'drp-srcf-cmp020',
         'drp-srcf-cmp021',
         'drp-srcf-cmp022',
         'drp-srcf-cmp023', # pepex
         'drp-srcf-cmp024',
         'drp-srcf-cmp046',
         'drp-srcf-cmp048',
         'drp-srcf-cmp050',]

if not args.readonly:
    with Pool(processes=None) as pool:
        result = pool.map_async(launch, nodes)
        result.wait()

from prettytable import PrettyTable
import glob

# Tabulate the results
table = PrettyTable()
table.field_names = ['HOST','DEV','LANE','BER']
fnames = glob.glob(f'{on}/*.dat')
fnames.sort()
data = []
lber = []
for fname in fnames:
    words = fname.split('.')
    host = words[0].split('/')[-1]
    datadev = words[1].split('_')[1]
    lane = words[2]
    f = open(fname,'r')
    ber = float(f.read().split(':')[1])
    lber.append(ber)
    sber = f'\033[0;33;40m{ber:.2g}\033[0m' if ber > args.limit else f'{ber:.2g}'
    data.append([host,datadev,lane,sber])

for i,row in enumerate(data):
    table.add_row(row,divider = (i+1)<len(data) and row[0]!=data[i+1][0])

print(table)

# Histogram the BER
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(tight_layout=True)
ax.hist(np.log10(lber), bins=40)
plt.xlabel('Log10 BER')
plt.savefig(f'{on}/all-hsd-drp.png')
plt.show()
