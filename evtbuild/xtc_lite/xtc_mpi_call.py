import subprocess
import time



file_name = '../hdf5/test_results/xtc_two_nodes_1_stripe.txt'
node_list = ['drp-tst-acc05','drp-tst-acc06']

nodes = ','.join(node_list)

sub_call = '`which mpirun` -q -map-by node --oversubscribe -n %i -H '+ nodes + ' python rwc_xtc_mpi.py | tee -a ' + file_name



for i in range(5):
    for core in range(1,17,2):
        tot_cores = core*len(node_list)

        out_call = sub_call % (tot_cores)
        # out_print = 'Calling %i total cores, %i cores per node' % (tot_cores, core)
        out_print = 'Calling %i cores' % tot_cores
        print(out_print)
        print(out_call)
        subprocess.call('echo %s >> %s' % (out_print, file_name), shell=True) 
        subprocess.call(out_call, shell=True)

