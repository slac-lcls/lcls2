from psana.parallelreader import ParallelReader
import os, glob

tmp_dir = './.tmp'
xtc_files = glob.glob(os.path.join(tmp_dir, '*r0001-e0*.xtc2'))
fds = [os.open(xtc_file, os.O_RDONLY) for xtc_file in xtc_files]

prl_reader = ParallelReader(fds)
prl_reader.read()
block = prl_reader.get_block()
assert memoryview(block).shape[0] == 2097164
