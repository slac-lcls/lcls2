from psana import DataSource

# a small utility for taking a command line datasource string
# like "exp=rixx43518,run=341" or "files=<filename>" and returning
# a DataSource object. Used by utilities like "detnames".

def DataSourceFromString(dsstring):
  if '=' in dsstring:
    if ':' in dsstring:
      print('Error: DataSource fields in psana2 must be split with "," not ":"')
      sys.exit(-1)

    # experiment/run specified, or shmem
    ds_split = dsstring.split(',')
    kwargs = {}
    for kwarg in ds_split:
      kwarg_split = kwarg.split('=')
      try:
        # see if it's a run number
        val=int(kwarg_split[1])
      except ValueError:
        val = kwarg_split[1]
      kwargs[kwarg_split[0]] = val
    return DataSource(**kwargs)
  else:
    # filename specified
    return DataSource(files=dsstring)
