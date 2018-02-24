import subprocess
import time

# Use snek environment on RHEL7
# snake-RHEL6 on RHEL6

file_name = 'test_results/out_two_node.txt'

sub_call = '`which mpirun` -q -map-by node --oversubscribe -n %i -H drp-tst-acc01,drp-tst-acc02,drp-tst-acc03,drp-tst-acc04,drp-tst-acc05,drp-tst-acc06 python rwc_mpi.py'# | tee -a out_6nodes3.txt'

node_list = ['drp-tst-acc01','drp-tst-acc02','drp-tst-acc03']#,'drp-tst-acc04','drp-tst-acc05','drp-tst-acc06']#,'daq-tst-dev02','daq-tst-dev05','daq-tst-dev04']

# 1/23/18 NVME test

#node_list = ['drp-tst-acc01', 'drp-tst-acc02','drp-tst-acc03','drp-tst-acc04']#,'daq-tst-dev01', 'daq-tst-dev02']

node_list = ['drp-tst-acc05', 'drp-tst-acc06']

nodes = ','.join(node_list)

sub_call = '`which mpirun` -q -map-by node --oversubscribe -n %i -H '+ nodes + ' python rwc_mpi.py | tee -a ' + file_name


for i in range(1):
    for cores in [4]:
        tot_cores = cores*len(node_list)
        out_call = sub_call % tot_cores
        out_print = 'Calling %i cores' % tot_cores
        print(out_call)
        subprocess.call('echo %s | tee -a %s' % (out_print, file_name), shell=True)
        subprocess.call(out_call, shell=True)


# #node_list = ['drp-tst-acc01','drp-tst-acc02','drp-tst-acc03']
# node_list = ['daq-tst-dev02','daq-tst-dev03','daq-tst-dev04']
# nodes = ','.join(node_list)

# sub_call = '`which mpirun` -q -map-by node --oversubscribe -n %xi -H '+ nodes + ' python rwc_mpi.py | tee -a ' + file_name


# for i in range(3):
#     for cores in [6]:#,4,6,8,10,12]:
#         tot_cores = cores*len(node_list)
#         out_call = sub_call % tot_cores
#         out_print = 'Calling %i cores' % tot_cores
#         print(out_call)
#         subprocess.call('echo %s | tee -a %s' % (out_print, file_name), shell=True)
#         subprocess.call(out_call, shell=True)

