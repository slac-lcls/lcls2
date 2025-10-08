
import logging
logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d %(filename)s: %(message)s', level=logging.DEBUG)

from psana import DataSource

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
  parser.add_argument("dsname", help="psana datasource experiment/run (e.g. exp=xppd7114,run=43) or xtc2 filename or shmem='my_shmem_identifier'")
  parser.add_argument('-r','--raw', dest='raw', action='store_true')
  parser.add_argument('-e','--epics', dest='epics', action='store_true')
  parser.add_argument('-s','--scan', dest='scan', action='store_true')
  parser.add_argument('-i','--ids', dest='ids', action='store_true')
  args = parser.parse_args()

  if '=' in args.dsname:
    if ':' in args.dsname:
      print('Error: DataSource fields in psana2 must be split with "," not ":"')
      sys.exit(-1)

    # experiment/run specified, or shmem
    ds_split = args.dsname.split(',')
    kwargs = {}
    for kwarg in ds_split:
      kwarg_split = kwarg.split('=')
      try:
        # see if it's a run number
        val=int(kwarg_split[1])
      except ValueError:
        val = kwarg_split[1]
      kwargs[kwarg_split[0]] = val

    print('XXX kwargs:', kwargs)
    ds = DataSource(**kwargs)
    print('XXX DataSource is created')
  else:
    # filename specified
    ds = DataSource(files=args.dsname)

  print('XXX before next(ds.runs())' + 100*'=')
  myrun = next(ds.runs())
  print('XXX after next(ds.runs())' + 100*'=')


  if args.raw:
    headers = ['Name','Det Type','Data Type','Version']
    format_string = '{0:%d} | {1:%d} | {2:%d} | {3:%d}'
    names = myrun.xtcinfo
  elif args.epics:
    headers = ['Name','Data Type']
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

    print('XXX', names)
    for name in names:
      datatype = name[1]
      data = getattr(myrun.Detector(name[0]),datatype)
      segments = ','.join([str(segid) for segid in data._sorted_segment_ids])
      newnames.append((name[0],datatype,segments,data._uniqueid))
    names = newnames
  else:
    headers = ['Name','Data Type']
    format_string = '{0:%d} | {1:%d}'
    names = myrun.detinfo.keys()
    print('XXX myrun.detinfo.keys()', myrun.detinfo.keys())

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


detnames()

# EOF
