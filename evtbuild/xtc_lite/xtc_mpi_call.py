import subprocess
import time
import glob


file_name = 'test_results/results.txt'
node_list = ['drp-tst-acc0%i' % x for x in [1,2]]
# node_list = ['drp-tst-oss10']
nodes = ','.join(node_list)

sub_call = '`which mpirun` -q -map-by node --oversubscribe -n %i -H ' + nodes + ' python rwc_xtc_mpi.py | tee -a ' + file_name
core_list =  [1,2,4,8,12,16]
core_list = core_list[::-1]
for i in range(1):
    for core in core_list:
        tot_cores = core*len(node_list)
        out_call = sub_call % (tot_cores)
        # out_print = 'Calling %i total cores, %i cores per node' % (tot_cores, core)
        out_print = 'Calling %i cores' % tot_cores
        print(out_print)
        print(out_call)
        subprocess.call('echo %s >> %s' % (out_print, file_name), shell=True)
        subprocess.call(out_call, shell=True)

