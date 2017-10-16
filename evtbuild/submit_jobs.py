import subprocess,time
import re, sys, os
import glob
from picklecreate import load_config

folder = str(sys.argv[1])
logdir = 'log_'+folder

try:
        os.mkdir(logdir)
except OSError:
        print('Directory %s already exists' % logdir)

# load the config dictionary

config_dict = load_config()
batch_sizes = config_dict['batches']*config_dict['num_repeats']
core_nums = config_dict['cores']

try:
       
    logs = []
    for batch in batch_sizes:
        for cores in core_nums:
            print('Submitting cores %i, batches %i' % (cores, batch)) 

            time.sleep(1)
            logname = '%s/%%J_core_%i_batch_%i.log' % (logdir, cores, batch)
            logs.append(logname)
            call_string = 'bsub -q psnehq -n %i -o %s mpirun python analysisScript.py %i %s' % (cores, logname, batch, folder)
            print('Calling %s' % call_string)
            job = subprocess.check_output(call_string, shell=True, stderr=subprocess.STDOUT)

            job_id = re.search('\<(\d+)\>', job)
            job_id = job_id.group(1)
            print('Job ID: %s' % job_id)
            #wait for the job to finish
            while True:
                out = subprocess.check_output('bjobs -d',shell=True, stderr=subprocess.STDOUT)
                if re.search(job_id, out):
                        print('Job %s finished' % job_id)
                        break
                else:
                        time.sleep(0.5)
#parse the logs and write out a summary
finally:
    run_data = []
#    print(glob.glob('%s/*' % logdir))
    for filename in glob.glob('%s/*' % logdir):
      #  print(filename)
        f = open(filename, 'r')
        logtxt = f.read()
        f.close()
        try:
            cores = re.search('(core_)(\d+)',filename).group(2)            
            batches = re.search('(batch_)(\d+)',filename).group(2)
            num_evts = re.search('(Number of events )(\d+)', logtxt).group(2)
            time_elapsed = re.search('(Time elapsed: )([\d.]+)', logtxt).group(2)
            file_size = re.search('(file size: )([\d.]+)', logtxt).group(2)
            average_speed = re.search('(Average speed: )([\d.]+)', logtxt).group(2)
            extr_data = [int(cores), int(batches), int(num_evts), time_elapsed, file_size, float(average_speed)]
           # print(extr_data)
            run_data.append(extr_data)
        except Exception as e:
            pass

    with open('core_output_%s.txt' % folder, 'w') as f:
        for line in run_data:
            for item in line:
                f.write(str(item)+'\t')
            f.write('\n')

    




    
