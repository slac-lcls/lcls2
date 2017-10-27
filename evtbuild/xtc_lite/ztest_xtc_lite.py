import subprocess

for i in range(1):
    for core_num in [1,2,4,8,10,12,16]:
        comm = 'mpirun -n %i python write_xtc_lite.py >> xtc_write_ztest.txt' % core_num
        print(comm)
        subprocess.call(comm, shell=True)


        comm2 = 'mpirun -n %i python read_xtc_lite.py >> xtc_read_ztest.txt' % core_num
        print(comm2)
        subprocess.call(comm2, shell=True)
