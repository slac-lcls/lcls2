#!/usr/bin/env python

cmd_seq = ['/bin/bash', '-l', '-c', '. /reg/g/psdm/etc/psconda.sh; echo "PATH: $PATH"; echo "CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"; detnames exp=xppx44719:run=11'] 
#cmd = '/bin/bash -l -c "source /reg/g/psdm/etc/psconda.sh; echo $PATH"' # works with shell=True

env1 = {'PATH':'/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/puppetlabs/bin'}
import subprocess
p = subprocess.Popen(cmd_seq, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, executable='/bin/bash', env=env1) # , shell=True, env=None,
text = p.communicate()[0].decode("utf-8")
print(text)
