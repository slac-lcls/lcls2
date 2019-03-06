from psana import DataSource

import argparse

def detnames():
  parser = argparse.ArgumentParser()
  parser.add_argument("dsname", help="psana datasource string (e.g. exp=xppd7114:run=43 or filename)")
  parser.add_argument('-r','--raw', dest='raw', action='store_true')
  args = parser.parse_args()

  ds = DataSource(args.dsname)
  myrun = next(ds.runs())

  if args.raw:
    headers = ['Name','Det Type','Data Type','Version']
    format_string = '{0:%d} | {1:%d} | {2:%d} | {3:%d}'
    names = myrun.xtcinfo
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
