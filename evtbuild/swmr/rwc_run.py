import subprocess, glob, os, time

out_write_call = "`which mpirun` -q --oversubscribe -n %i -H drp-tst-acc05,drp-tst-acc06 swmr_write.sh > swmr_write_out.txt"

out_filter_call = "`which mpirun` -q --oversubscribe -n %i -H drp-tst-acc03,drp-tst-acc04 swmr_read.sh > swmr_filter_out.txt"

out_copy_call = "`which mpirun` -q --oversubscribe -n %i -H drp-tst-acc05,drp-tst-acc06 swmr_read_nersc.sh > swmr_copy_out.txt"


cores = [1,2,4,8,16,31]
cores = [31]

def clear_files(path):
    files = glob.glob(path)
    for fil in files:
        os.remove(fil)

    #check if the files have actually been deleted
    while True:
        files = glob.glob(path)
        if len(files) == 0:
            break
        else:
            time.sleep(0.2)
            
for core in cores:
    clear_files('/drpffb/eliseo/data/swmr/*.h5')
    print('Writing %i cores' % core)
    writ = subprocess.Popen(out_write_call % core, shell=True)

    read_cores = core + 1
    print('Filtering %i cores' % core)
    filt = subprocess.Popen(out_filter_call % read_cores, shell=True)

  #  print('Copying %i cores' % core)
   # copy = subprocess.Popen(out_copy_call % read_cores, shell=True)

    writ.wait()
    filt.wait()
#    copy.wait()

    print('Done with %i cores' % core)
  
 
        

        
