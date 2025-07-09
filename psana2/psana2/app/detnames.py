from psana2 import DataSource
from psana2.psexp.utils import DataSourceFromString

import argparse
import sys

def detnames():

  # this argument parsing doesn't feel ideal.  ideally user should
  # be able to pass in the "python code" specifying the datasource
  # on the command line (e.g. exp=xpptut15,run=[1,2,4,7] or
  # files=['data1.xtc2', 'data2.xtc2'].  But having the shell not mangle
  # that and then parsing it appropriately feels challenging.  So only
  # support single runs, shmem, and filenames using standard sys.argv.

  parser = argparse.ArgumentParser()
  parser.add_argument("dsname", help="psana datasource experiment/run (e.g. exp=xppd7114,run=43) or xtc2 filename or shmem=<my_shmem_identifier>")
  parser.add_argument('-e','--epics', dest='epics', action='store_true', help='Dump epics variable aliases for use with Detector interface')
  parser.add_argument('-s','--scan', dest='scan', action='store_true', help='Dump step-scan names for use with Detector interface')
  parser.add_argument('-r','--raw', dest='raw', action='store_true', help='Expert only: dump data types in raw data')
  parser.add_argument('-i','--ids', dest='ids', action='store_true', help="Expert only: dump segment-id's and unique-id's used by calibration database")
  args = parser.parse_args()

  ds = DataSourceFromString(args.dsname)

  myrun = next(ds.runs())
  if args.raw:
    headers = ['Name','Det Type','Data Type','Version']
    format_string = '{0:%d} | {1:%d} | {2:%d} | {3:%d}'
    names = myrun.xtcinfo
  elif args.epics:
    headers = ['Detector Name','Epics Name']
    format_string = '{0:%d} | {1:%d}'
    names = myrun.epicsinfo
  elif args.scan:
    headers = ['Name','Data Type']
    format_string = '{0:%d} | {1:%d}'
    names = myrun.scaninfo
  elif args.ids:
    headers = ['Name','Data Type','Segments','UniqueId']
    format_string = '{0:%d} | {1:%d} | {2:%d} | {3:%d}'
    names = myrun.detinfo.keys()
    newnames = []
    for name in names:
      datatype = name[1]
      data = getattr(myrun.Detector(name[0]),datatype)
      segments = ','.join([str(segid) for segid in data._sorted_segment_inds])
      newnames.append((name[0],datatype,segments,data._uniqueid))
    names = newnames
  else:
    headers = ['Name','Data Type']
    format_string = '{0:%d} | {1:%d}'
    names = myrun.detinfo.keys()

  maxlen = [len(h) for h in headers]
  for ntuple in names:
    lengths = [len(n) for n in ntuple]
    maxlen = [max(oldmax,length) for oldmax,length in zip(maxlen,lengths)]

  # assumes that data rows are tuples
  template = format_string % tuple(maxlen)
  header = template.format(*headers)
  print('-'*len(header))
  print(header)
  print('-'*len(header))
  for n in names:
    print(template.format(*n))
  print('-'*len(header))

