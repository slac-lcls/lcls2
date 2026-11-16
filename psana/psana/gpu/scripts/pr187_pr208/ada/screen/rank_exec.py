"""Launch sanitizer only on GPU BD ranks in diagnostic MPI processes."""
import os,sys
command=[sys.executable,'-B','-u']+sys.argv[1:]
if int(os.environ['OMPI_COMM_WORLD_RANK'])>=2 and os.environ.get('BENCH_SANITIZER_LOG'):
 command=['/sdf/group/lcls/ds/tools/cuda12.8/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','99','--log-file',os.environ['BENCH_SANITIZER_LOG']]+command
os.execv(command[0],command)
