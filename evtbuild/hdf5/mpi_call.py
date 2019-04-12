import subprocess
import time

# Use snek environment on RHEL7
# snake-RHEL6 on RHEL6



file_name = 'test_results/results.txt'
node_list = ['drp-tst-acc0%i' % x for x in [1,2,3,4]]
nodes = ','.join(node_list)


sub_call = '`which mpirun` -q -map-by node --oversubscribe -n %i -H ' + nodes + ' python rwc_mpi.py | tee -a ' + file_name

core_list =  [8]
core_list = core_list[::-1]
for i in range(1):
    for core in core_list:
        tot_cores = core*len(node_list)
        out_call = sub_call % tot_cores
        out_print = 'Calling %i cores' % tot_cores
        print(out_call)
        subprocess.call('echo %s | tee -a %s' % (out_print, file_name), shell=True)
        subprocess.call(out_call, shell=True)

