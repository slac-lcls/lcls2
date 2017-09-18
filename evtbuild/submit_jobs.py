import subprocess,time
import re, sys, os
import glob

batch_sizes = [10,50,100,500]
batch_sizes = [5]
core_nums=[2,4,8,16,32,64]*4

#batch_sizes = [100]
#core_nums = [64]
folder = str(sys.argv[1])
logdir = 'log_'+folder

try:
        os.mkdir(logdir)
except OSError:
        print('Directory %s already exists' % logdir)


try:
       
    logs = []
    for batch in batch_sizes:
        for cores in core_nums:
            print('Submitting cores %i, batches %i' % (cores, batch)) 

            time.sleep(1)
            logname = '%s/%%J_core_%i_batch_%i.log' % (logdir, cores, batch)
            logs.append(logname)
            call_string = 'bsub -q psnehq -n %i -o %s mpirun python analysisScript2.py %i %s' % (cores, logname, batch, folder)
            print('Calling %s' % call_string)
            subprocess.call(call_string, shell=True)

            
            while True:
#                break
                out = subprocess.check_output('bjobs', stderr=subprocess.STDOUT)
                if out == 'No unfinished job found\n':
                    time.sleep(1)
                else:
                    print(out)
                    print('Job started')
                    break
            time.sleep(1)
            while True:
#                break
                out = subprocess.check_output('bjobs', stderr=subprocess.STDOUT)
                if out == 'No unfinished job found\n':
                    print(out)
                    print('Job done')
                    break
                else:
                    time.sleep(1)


    while True:
        break
        out = subprocess.check_output('bjobs', stderr=subprocess.STDOUT)
        if out == 'No unfinished job found\n':
            break
        else:
            time.sleep(1)

finally:
    run_data = []
    for filename in glob.glob('%s/*' % logdir):
      #  print(filename)
        f = open(filename, 'r')
        logtxt = f.read()
        f.close()
        try:
            cores = re.search('(core_)(\d+)',filename).group(2)
            batches = re.search('(batch_)(\d+)',filename).group(2)
            num_evts = re.search('(Number of events )(\d+)', logtxt).group(2)
            time_elapsed = re.search('(Time elapsed: )(\d+)', logtxt).group(2)
            file_size = re.search('(File size: )([\d+.]*)', logtxt).group(2)
            average_speed = re.search('(Average speed: )([\d+.]*)', logtxt).group(2)
            extr_data = [int(cores), int(batches), int(num_evts), int(time_elapsed), int(file_size), float(average_speed)]
            print(extr_data)
            run_data.append(extr_data)
        except AttributeError:
            pass

    with open('core_output_%s.txt' % folder, 'w') as f:
        for line in run_data:
            for item in line:
                f.write(str(item)+'\t')
            f.write('\n')

    




    
