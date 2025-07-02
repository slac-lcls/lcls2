import sys

from psana import DataSource

# a small utility for taking a command line datasource string
# like "exp=rixx43518,run=341" or "files=<filename>" and returning
# a DataSource object. Used by utilities like "detnames".


def datasource_kwargs_from_string(dsstring, detname=None):
    """
    Accepts (str) dsstring as
    0. xtc file name,
    1. string of comma-separated (no spaces) simple parameters, ex:
       "file=<fname.xtc>,exp=<expname>,run=<runs>,dir=<xtc-dir>, ...",
    2. dict (string in pythonic syntaxes) of generic parameters, e.g.:
       "{'exp':'tmoc00318', 'run':[10,11,12], 'dir':'/a/b/c/', 'detectors':['epicsinfo', 'tmo_opal1', 'ebeam']}"
    Returns (dict) of kwargs for DataSource(**kwargs)

    See: https://confluence.slac.stanford.edu/display/LCLSIIData/psana

    files: str - xtc2 file name
    exp: str - experiment name
    run: int run number or str with comma-separated run numbers, list of runs ???? THIS WOULD NOT WORK
    dir: str - xtc2 directory name
    max_events: int - maximal number of events to process
    live: True
    timestamp = np.array([4194783241933859761,4194783249723600225,4194783254218190609,4194783258712780993], dtype=np.uint64)??? list of ints?
    intg_det = 'andor'
    batch_size = 1
    detectors = ['epicsinfo', 'tmo_opal1', 'ebeam'] - only reads these detectors (faster)  ???? THIS WOULD NOT WORK
    smd_callback= smd_callback,                     - smalldata callback (see notes above)
    small_xtc   = ['tmo_opal1'],                    - detectors to be used in smalldata callback ???? THIS WOULD NOT WORK
    shmem='tmo' or 'rix',...
    """
    if dsstring.lstrip()[0] == "{":
        return eval(dsstring)

    kwargs = {}
    if "=" in dsstring:
        if ":" in dsstring:
            print('Error: DataSource fields in psana2 must be split with "," not ":"')
            sys.exit(-1)
        # experiment/run specified, or shmem
        for kwarg in dsstring.split(","):
            k, v = kwarg.split("=")
            val = (
                str(v)
                if k in ("files", "exp", "dir", "shmem")
                else (
                    int(v)
                    if k == "run" and ("," not in v)
                    else (
                        int(v)
                        if k in ("max_events", "batch_size")
                        else v == "True"
                        if k in ("live",)
                        else None
                    )
                )
            )
            if val is None:
                try:
                    # see if it's a run number
                    val = int(v)
                except ValueError:
                    val = v
            kwargs[k] = val
        #return kwargs
    else:
        # filename specified
        kwargs["files"] = dsstring

    if detname is not None: kwargs["detectors"] = [detname,]
    return kwargs

def datasource_kwargs_to_string(**kwargs):
    """returns string presentation for dict of DataSource kwargs"""
    return ",".join(["%s=%s" % (k, str(v)) for k, v in kwargs.items()])


def DataSourceFromString(dsstring):
    return DataSource(**datasource_kwargs_from_string(dsstring))


# EOF
