from psana.parallelreader import ParallelReader
from psana.psexp.packet_footer import PacketFooter
import os, glob
from psana import dgram

tmp_dir = './.tmp'
xtc_files = [os.path.join(tmp_dir, 'data-r0001-e00.xtc2'), os.path.join(tmp_dir, 'data-r0001-e01.xtc2')]
fds = [os.open(xtc_file, os.O_RDONLY) for xtc_file in xtc_files]

configs = [dgram.Dgram(file_descriptor=fd) for fd in fds]

prl_reader = ParallelReader(fds)
block = prl_reader.get_block()

pf = PacketFooter(view=block)
views = pf.split_packets()

for i in range(len(views)):
    config, view = configs[i], views[i]
    d = dgram.Dgram(config=config, view=view)
    if i == 0: # first epics file
        assert getattr(d.xppepics[0].fuzzy, 'HX2:DVD:GCC:01:PMON') == 41.0
    elif i == 1: # second epics file
        assert getattr(d.xppepics[0].fuzzy, 'XPP:VARS:FLOAT:02') == 41.0

