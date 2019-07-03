from psana.parallelreader import ParallelReader
from psana.psexp.packet_footer import PacketFooter
import os, glob
from psana import dgram

tmp_dir = os.path.join(os.environ.get('TEST_XTC_DIR', os.getcwd()),'.tmp')
xtc_files = [os.path.join(tmp_dir, 'data-r0001-s00.xtc2'), os.path.join(tmp_dir, 'data-r0001-s01.xtc2')]
fds = [os.open(xtc_file, os.O_RDONLY) for xtc_file in xtc_files]

configs = [dgram.Dgram(file_descriptor=fd) for fd in fds]

prl_reader = ParallelReader(fds)
block = prl_reader.get_block()

pf = PacketFooter(view=block)
views = pf.split_packets()

for i in range(len(views)):
    config, view = configs[i], views[i]
    d = dgram.Dgram(config=config, view=view)
    #assert getattr(d.epics[0].fast, 'HX2:DVD:GCC:01:PMON') == 41.0
    #assert getattr(d.epics[0].slow, 'XPP:GON:MMS:01:RBV') == 41.0
