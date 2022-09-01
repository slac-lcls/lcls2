import sys, os
import time
import getopt
import sysv_ipc
import pprint

try:
    # doesn't exist on macos
    from shmem import PyShmemClient
except:
    pass
from psana import dgram
from psana.event import Event
from psana.detector import detectors
from psana.psexp.event_manager import TransitionId
import numpy as np

def dumpDict(dict,indent):
    for k in sorted(dict.keys()):
        if hasattr(dict[k],'__dict__'):
            print(' '*indent,k)
            dumpDict(dict[k].__dict__,indent+2)
        else:
            print(' '*indent,k,dict[k])

# method to dump dgrams to stdout.  ideally this would move into dgram.cc
def dumpDgram(d):
    dumpDict(d.__dict__,0)

# Maximum length of filenames
FN_L = 200
# Maximum no. of shmem connection retries
SHMEM_CONN_MAX_RETRIES = 100

# Warning: If XtcData::Dgram ever changes, this function will likely need to change
def _service(view):
    iSvc = 2                    # Index of service field, in units of uint32_t
    return (np.array(view, copy=False).view(dtype=np.uint32)[iSvc] >> 24) & 0x0f

# Warning: If XtcData::Dgram ever changes, this function will likely need to change
def dgSize(view):
    iExt = 5                    # Index of extent field, in units of uint32_t
    txSize = 3 * 4              # sizeof(XtcData::TransitionBase)
    return txSize + np.array(view, copy=False).view(dtype=np.uint32)[iExt]

class DgramManager(object):

    def __init__(self, xtc_files, configs=[], fds=[],
            tag=None, run=None, max_retries=0,
            config_consumers=[]):
        """ Opens xtc_files and stores configs.
        If file descriptors (fds) is given, reuse the given file descriptors.
        """
        self.xtc_files = []
        self.shmem_cli = None
        self.mq_recv = None
        self.mq_send = None
        self.shm_recv = None
        self.shm_send = None
        self.shm_recv_mv = None
        self.shm_send_mv = None
        self.shm_size = None
        self.shmem_kwargs = {'index':-1,'size':0,'cli_cptr':None}
        self.configs = []
        self._timestamps = [] # built when iterating
        self._run = run
        self.found_endrun = True

        # We check for EndRun when we hit the end of RunSingleFile and RunShmem.
        # If there's no EndRun, a fake one will be created. In this case,
        # the BeginRun will be saved in the buffered_beginruns, so that
        # reading can continue with the new but same BeginRun.
        self.buffered_beginruns = []

        self.max_retries = max_retries
        self.chunk_ids = []
        self.config_consumers = config_consumers
        self.tag = tag

        if isinstance(xtc_files, (str)):
            self.xtc_files = np.array([xtc_files], dtype='U%s'%FN_L)
        elif isinstance(xtc_files, (list, np.ndarray)):
            if len(xtc_files) > 0: # handles smalldata-only case
                if xtc_files[0] == 'shmem':
                    view = self._connect_shmem_cli(self.tag)
                    d = dgram.Dgram(view=view)
                    #self.configs += [d]
                    # The above line is kept to note that prior to the change below,
                    # the configs are saved as a list. Note that only the most recent
                    # one is used. Mona changed this to "replace" so at a time, there's
                    # only one config.
                    self.set_configs([d])
                elif xtc_files[0] == 'drp':
                    view = self._connect_drp()
                    d = dgram.Dgram(view=view)
                    self.set_configs([d])
                else:
                    self.xtc_files = np.asarray(xtc_files, dtype='U%s'%FN_L)


        self.given_fds = True if len(fds) > 0 else False
        if self.given_fds:
            self.fds = np.asarray(fds, dtype=np.int32)
        else:
            self.fds = np.array([os.open(xtc_file, os.O_RDONLY) for xtc_file in self.xtc_files], dtype=np.int32)

        self.fds_map = {}
        for fd, xtc_file in zip(self.fds, self.xtc_files):
            self.fds_map[fd] = xtc_file

        given_configs = True if len(configs) > 0 else False
        if given_configs:
            self.set_configs(configs)
        elif xtc_files[0] != 'shmem' and xtc_files[0] != 'drp':
            self.set_configs([dgram.Dgram(file_descriptor=fd, max_retries=self.max_retries) for fd in self.fds])

        self.calibconst = {} # initialize to empty dict - will be populated by run class
        self.n_files = len(self.xtc_files)
        self.set_chunk_ids()


    def _connect_shmem_cli(self, tag):
        # ShmemClients open a connection in connect() and close it in
        # the destructor. By creating a new client every time, we ensure
        # that python call the destructor in the gc routine.
        self.shmem_cli = PyShmemClient()
        for retry in range(SHMEM_CONN_MAX_RETRIES):
            #establish connection to available server - blocking
            status = int(self.shmem_cli.connect(tag,0))
            if status == 0:
                break
            time.sleep(0.01)
        assert not status,'shmem connect failure %d' % status
        #wait for first configure datagram - blocking
        view = self.shmem_cli.get(self.shmem_kwargs)
        assert view
        # Release shmem buffer after copying Transition data
        # cpo: copy L1Accepts too because some shmem
        # applications like AMI's pickN can hold references
        # to dgrams for a long time, consuming the shmem buffers
        # and creating a deadlock situation. could revisit this
        # later and only deep-copy arrays inside pickN, for example
        # but would be more fragile.
        barray = bytes(view[:dgSize(view)])
        self.shmem_cli.freeByIndex(self.shmem_kwargs['index'], self.shmem_kwargs['size'])
        view = memoryview(barray)
        return view

    def _connect_drp(self):
        try:
            self.mq_recv = sysv_ipc.MessageQueue(200000)
            self.mq_send = sysv_ipc.MessageQueue(200001)
        except sysv_ipc.Error as exp:
            assert(False)
        try:
            self.shm_recv = sysv_ipc.SharedMemory(200002, size=40000000)
            self.shm_send = sysv_ipc.SharedMemory(200003, size=40000000)
        except sysv_ipc.Error as exp:
            assert(False)

        self.shm_recv_mv = memoryview(self.shm_recv)
        self.shm_send_mv = memoryview(self.shm_send)

        self.mq_send.send(b"g")
        message, priority = self.mq_recv.receive()
        barray = bytes(self.shm_recv_mv[:])
        view = memoryview(barray)
        self.shm_size = view.nbytes   
        return view

    def set_configs(self, dgrams):
        """Save and setup given dgrams class configs."""
        self.configs = dgrams
        self._setup_det_class_table()
        self._set_configinfo()

    def _setup_det_class_table(self):
        """
        this function gets the version number for a (det, drp_class) combo
        maps (dettype,software,version) to associated python class and
        detector info for a det_name maps to dettype, detid tuple.
        """
        det_classes = {'epics': {}, 'scan': {}, 'step': {}, 'normal': {}}

        xtc_info = []
        det_info_table = {}

        # collect corresponding stream id for a detector (first found)
        det_stream_id_table = {}

        # loop over the dgrams in the configuration
        # if a detector/drp_class combo exists in two cfg dgrams
        # it will be OK... they should give the same final Detector class
        for i, cfg_dgram in enumerate(self.configs):
            for det_name, det_dict in cfg_dgram.software.__dict__.items():
                # go find the class of the first segment in the dict
                # they should all be identical
                first_key = next(iter(det_dict.keys()))
                det = det_dict[first_key]

                if det_name not in det_classes:
                    det_class_table = det_classes['normal']
                else:
                    det_class_table = det_classes[det_name]

                dettype, detid = (None, None)
                for drp_class_name, drp_class in det.__dict__.items():

                    # collect detname maps to dettype and detid
                    if drp_class_name == 'dettype':
                        dettype = drp_class
                        continue

                    if drp_class_name == 'detid':
                        detid = drp_class
                        continue

                    # FIXME: we want to skip '_'-prefixed drp_classes
                    #        but this needs to be fixed upstream
                    if drp_class_name.startswith('_'): continue

                    # use this info to look up the desired Detector class
                    versionstring = [str(v) for v in drp_class.version]
                    class_name = '_'.join([det.dettype, drp_class.software] + versionstring)
                    xtc_entry = (det_name,det.dettype,drp_class_name,'_'.join(versionstring))
                    if xtc_entry not in xtc_info:
                        xtc_info.append(xtc_entry)
                    if hasattr(detectors, class_name):
                        DetectorClass = getattr(detectors, class_name) # return the class object
                        det_class_table[(det_name, drp_class_name)] = DetectorClass
                    else:
                        pass

                det_info_table[det_name] = (dettype, detid)

                if det_name not in det_stream_id_table:
                    det_stream_id_table[det_name] = i

        # Add products of this function to itself and the consumers
        for config_consumer in [self]+self.config_consumers:
            setattr(config_consumer, 'det_classes', det_classes)
            setattr(config_consumer, 'xtc_info', xtc_info)
            setattr(config_consumer, 'det_info_table', det_info_table)
            setattr(config_consumer, 'det_stream_id_table', det_stream_id_table)

    def _set_configinfo(self):
        """ From configs, we generate a dictionary lookup with det_name as a key.
        The information stored the value field contains:

        - configs specific to that detector
        - sorted_segment_ids
          used by Detector cls for checking if an event has correct no. of segments
        - detid_dict
          has segment_id as a key
        - dettype
        - uniqueid
        """
        configinfo_dict = {}

        for detcls_name, det_class in self.det_classes.items(): # det_class is either normal or envstore ('epics', 'scan', 'step')
            for (det_name, _), _ in det_class.items():
                # we lose a "one-to-one" correspondence with event dgrams.  we may have
                # to put in None placeholders at some point? - mona and cpo
                det_configs = [cfg for cfg in self.configs if hasattr(cfg.software, det_name)]
                sorted_segment_ids = []
                # a dictionary of the ids (a.k.a. serial-number) of each segment
                detid_dict = {}
                dettype = ""
                uniqueid = ""
                for config in det_configs:
                    seg_dict = getattr(config.software, det_name)
                    sorted_segment_ids += list(seg_dict.keys())
                    for segment, det in seg_dict.items():
                        detid_dict[segment] = det.detid
                        dettype = det.dettype

                sorted_segment_ids.sort()

                uniqueid = dettype
                for segid in sorted_segment_ids:
                    uniqueid += '_'+detid_dict[segid]

                configinfo_dict[det_name] = type("ConfigInfo", (), {\
                        "configs": det_configs, \
                        "sorted_segment_ids": sorted_segment_ids, \
                        "detid_dict": detid_dict, \
                        "dettype": dettype, \
                        "uniqueid": uniqueid})

        for config_consumer in [self]+self.config_consumers:
            setattr(config_consumer, 'configinfo_dict', configinfo_dict)

    def set_chunk_ids(self):
        """ Generates a list of chunk ids for all stream files

        Chunk Id is extracted from data file name:
        New format: xpptut15-r0001-s000-c000[.smd].xtc2
        Old format: xpptut15-r0001-s00-c00[.smd].xtc2
        """
        if len(self.xtc_files) == 0: return
        if self.xtc_files[0] == 'shmem': return
        for xtc_file in self.xtc_files:
            filename = os.path.basename(xtc_file)
            st = filename.find('-c')
            en = filename.find('.smd.xtc2')
            if en < 0:
                en = filename.find('.xtc2')
            if st >= 0 and en >= 0:
                self.chunk_ids.append(int(filename[st+2:en]))

    def get_chunk_id(self, ind):
        if not self.chunk_ids: return None
        return self.chunk_ids[ind]

    def set_chunk_id(self, ind, new_chunk_id):
        self.chunk_ids[ind] = new_chunk_id

    def close(self):
        if not self.given_fds:
            for fd in self.fds:
                os.close(fd)

    def __iter__(self):
        return self

    def _check_missing_endrun(self, beginruns=None):
        fake_endruns = None
        if not self.found_endrun: # there's no previous EndRun
            sec = (self._timestamps[-1] >> 32) & 0xffffffff
            usec = int((self._timestamps[-1] & 0xffffffff) * 1e3 + 1)
            if beginruns:
                self.buffered_beginruns = [dgram.Dgram(config=config,
                        view=d, offset=0, size=d._size)
                        for d, config in zip(beginruns, self.configs)]
            fake_endruns = [dgram.Dgram(config=config, fake_endrun=1, \
                    fake_endrun_sec=sec, fake_endrun_usec=usec) \
                    for config in self.configs]
            self.found_endrun = True
        else:
            self.found_endrun = False
        return fake_endruns

    def __next__(self):
        """ only support sequential read - no event building"""
        if self.buffered_beginruns:
            self.found_endrun = False
            evt = Event(self.buffered_beginruns, run=self._run)
            self._timestamps += [evt.timestamp]
            self.buffered_beginruns = []
            return evt

        if self.shmem_cli:
            view = self.shmem_cli.get(self.shmem_kwargs)
            if view:
                # Release shmem buffer after copying Transition data
                # cpo: copy L1Accepts too because some shmem
                # applications like AMI's pickN can hold references
                # to dgrams for a long time, consuming the shmem buffers
                # and creating a deadlock situation. could revisit this
                # later and only deep-copy arrays inside pickN, for example
                # but would be more fragile.
                barray = bytes(view[:dgSize(view)])
                self.shmem_cli.freeByIndex(self.shmem_kwargs['index'], self.shmem_kwargs['size'])
                view = memoryview(barray)
                # use the most recent configure datagram
                config = self.configs[len(self.configs)-1]
                d = dgram.Dgram(config=config,view=view)
            else:
                view = self._connect_shmem_cli(self.tag)
                config = self.configs[len(self.configs)-1]
                d = dgram.Dgram(config=config,view=view)
                if d.service() == TransitionId.Configure:
                    self.set_configs([d])
                else:
                    raise RuntimeError(f"Configure expected, got {d.service()}")
            dgrams = [d]
        elif self.mq_recv:
            self.mq_send.send(b"g")
            message, priority = self.mq_recv.receive()
            if message == b"g":
                # use the most recent configure datagram
                config = self.configs[len(self.configs)-1]
                d = dgram.Dgram(config=self.configs[-1], view=self.shm_recv_mv)
            else:
                view = self._connect_drp()
                config = self.configs[len(self.configs)-1]
                d = dgram.Dgram(config=config,view=view)
                if d.service() == TransitionId.Configure:
                    self.set_configs([d])
                else:
                    raise RuntimeError(f"Configure expected, got {d.service()}")                
            dgrams = [d]    
        else:
            try:
                dgrams = [dgram.Dgram(config=config, max_retries=self.max_retries) for config in self.configs]
            except StopIteration as err:
                fake_endruns = self._check_missing_endrun()
                if fake_endruns:
                    dgrams = fake_endruns
                else:
                    print(err)
                    raise StopIteration

        # Check BeginRun - EndRun pairing
        service = dgrams[0].service()
        if service == TransitionId.BeginRun:
            fake_endruns = self._check_missing_endrun(beginruns=dgrams)
            if fake_endruns:
                dgrams = fake_endruns

        if service == TransitionId.EndRun:
            self.found_endrun = True

        if service == TransitionId.Configure:
            self.set_configs(dgrams)
            return self.__next__()

        evt = Event(dgrams, run=self.get_run())
        self._timestamps += [evt.timestamp]
        return evt

    def jumps(self, dgram_i, offset, size):
        if offset == 0 and size == 0:
            d = None
        else:
            try:
                d = dgram.Dgram(file_descriptor=self.fds[dgram_i],
                    config=self.configs[dgram_i],
                    offset=offset,
                    size=size,
                    max_retries=self.max_retries)
            except StopIteration:
                d = None
        return d

    def jump(self, offsets, sizes):
        """ Jumps to the offset and reads out dgram on each xtc file.
        This is used in normal mode (multiple detectors with MPI).
        """
        assert len(offsets) > 0 and len(sizes) > 0
        dgrams = [self.jumps(dgram_i, offset, size) for dgram_i, (offset, size)
            in enumerate(zip(offsets, sizes))]
        evt = Event(dgrams, run=self._run)
        return evt

    def get_timestamps(self):
        return np.asarray(self._timestamps, dtype=np.uint64) # return numpy array for easy search later

    def set_run(self, run):
        self._run = run

    def get_run(self):
        return self._run

def parse_command_line():
    opts, args_proper = getopt.getopt(sys.argv[1:], 'hvd:f:')
    xtcdata_filename="data.xtc"
    for option, parameter in opts:
        if option=='-h': usage_error()
        if option=='-f': xtcdata_filename = parameter
    if xtcdata_filename is None:
        xtcdata_filename="data.xtc"
    return (args_proper, xtcdata_filename)

def getMemUsage():
    pid=os.getpid()
    ppid=os.getppid()
    cmd="/usr/bin/ps -q %d --no-headers -eo size" % pid
    p=os.popen(cmd)
    size=int(p.read())
    return size

def main():
    args_proper, xtcdata_filename = parse_command_line()
    ds=DgramManager(xtcdata_filename)
    print("vars(ds):")
    for var_name in sorted(vars(ds)):
        print("  %s:" % var_name)
        e=getattr(ds, var_name)
        if not isinstance(e, (tuple, list, int, float, str)):
            for key in sorted(e.__dict__.keys()):
                print("%s: %s" % (key, e.__dict__[key]))
    print()
    count=0
    for evt in ds:
        print("evt:", count)
        for dgram in evt:
            for var_name in sorted(vars(dgram)):
                val=getattr(dgram, var_name)
                print("  %s: %s" % (var_name, type(val)))
            a=dgram.xpphsd.raw.array0Pgp
            try:
                a[0][0]=999
            except ValueError:
                print("The dgram.xpphsd.raw.array0Pgp is read-only, as it should be.")
            else:
                print("Warning: the evt.array0_pgp array is writable")
            print()
        count+=1
    return

def usage_error():
    s="usage: python %s" %  os.path.basename(sys.argv[0])
    sys.stdout.write("%s [-h]\n" % s)
    sys.stdout.write("%s [-f xtcdata_filename]\n" % (" "*len(s)))
    sys.exit(1)

if __name__=='__main__':
    main()

