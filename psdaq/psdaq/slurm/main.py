####################################################################
# psbatch 
# Usage: psbatch
# Parameters:
#   - subcommand:   actions to be performed on the inputs
#   - inp_file:     input file with resources 
####################################################################

from typing import List
import json
import typer
import time
import asyncio
import psutil
import copy
import socket
from psdaq.slurm.utils import SbatchManager
from psdaq.slurm.subproc import SubprocHelper
import os, sys
from psdaq.slurm.config import Config
from IPython import embed


sbman = SbatchManager()
proc = SubprocHelper()
runner = None
LOCALHOST = socket.gethostname()

class Runner():
    def __init__(self, configfilename):
        # Allowing users' code to do relative 'import' in config file
        sys.path.append(os.path.dirname(configfilename))
        config_dict = {'platform': None, 'config': None}
        try:
            exec(compile(open(configfilename).read(), configfilename, 'exec'), {}, config_dict)
        except:
            print('Error parsing configuration file:', sys.exc_info()[1])
        self.platform = config_dict['platform']
        # Check if we are getting main or derived config file
        if config_dict['config'] is None:
            config = Config(config_dict['main_config'])
            self.config = config.main_config
        else:
            self.config = config_dict['config'].select_config
        # Find xpm number
        self.xpm_id = 99
        if 'control' in self.config:
            cmd_tokens = self.config['control']['cmd'].split()
            for cmd_index, cmd_token in enumerate(cmd_tokens):
                if cmd_token == '-x':
                    self.xpm_id = int(cmd_tokens[cmd_index+1])

        sbman.set_attr('platform', self.platform)


    def parse_config(self):
        """ Extract commands from the cnf file """
        use_feature = True
        for config_id, config_detail in self.config.items():
            if 'host' in config_detail:
                use_feature = False

        if use_feature:
            node_features = sbman.get_node_features()
        else:
            node_features = None
        
        data = {}
        for config_id, config_detail in self.config.items():
            config_detail['comment'] = sbman.get_comment(self.xpm_id, self.platform, config_id)
            if use_feature:
                found_node = None
                for node, features in node_features.items():
                    for feature, occupied in features.items():
                        if occupied: continue
                        if config_id == feature:
                            node_features[node][feature] = 1
                            found_node = node
                            break
                if not found_node:
                    node = "localhost"
                else:
                    node = found_node
            else:
                if 'host' in config_detail:
                    node = config_detail['host']
                else:
                    node = LOCALHOST
            if node not in data:
                job_details = {}
                job_details[config_id] = config_detail
                data[node] = job_details
            else:
                job_details = data[node]
                if config_id in job_details:
                    msg = f"Error: cannot create more than one {config_id} on {node}"
                    raise NameError(msg)
                else:
                    job_details[config_id] = config_detail
        self.sbjob = data
        self.node_features = node_features
        return 

    def list_jobs(self):
        cmd = ''
        user = os.environ.get('USER','')
        if not user: 
            print(f'Cannot list jobs for user. $USER variable is not set.')
        else:
            if sbman.as_step:
                cmd = "sacct --format=JobIDRaw,JobName%12,User,State,Start,Elapsed,NNodes,NodeList,Comment"
            else:
                cmd = f'squeue -u {user} -o "%10i %15j %8u %8T %20S %10M %6D %R %k"'
        cmd = f"xterm -fa 'Source Code Pro' -geometry 120x31+15+15 -e watch -n 5 --no-title '{cmd}'" 
        asyncio.run(proc.run(cmd))

    def submit(self):
        cmd = "sbatch << EOF\n"+sbman.sb_script+"\nEOF\n"
        asyncio.run(proc.run(cmd, wait_output=True))

def main(subcommand: str,
        cnf_file: str,
        as_step: bool = False,
        interactive: bool = False,
        ):
    global runner
    runner = Runner(cnf_file)
    runner.parse_config()
    sbman.as_step = as_step
    if subcommand == "start":
        start(interactive)
    elif subcommand == "status":
        ls(interactive=interactive)
    elif subcommand == "stop":
        stop()
    elif subcommand == "restart":
        restart()
    else:
        print(f'Unrecognized subcommand: {subcommand}')

def start(interactive):
    if sbman.as_step:
        sbman.generate_as_step(runner.sbjob, runner.node_features)
        runner.submit()
    else:
        for node, job_details in runner.sbjob.items():
            for job_name, details in job_details.items():
                #        continue
                sbman.generate(node, job_name, details, runner.node_features)
                runner.submit()
    if interactive: embed()

def ls(interactive=True):
    if runner is None: return
    if interactive:
        runner.list_jobs()
        embed()
    else:
        job_details = {}
        for i, job_info in enumerate(sbman.get_job_info(format_string='"%i %j %T %R %k"', noheader=True)):
            job_id, job_name, state, nodelist, comment = job_info.strip('"').split()
            job_details[comment] = {'job_id': job_id, 'job_name': job_name, 'state': state, 'nodelist': nodelist}

        print('%20s %12s %10s %40s' %('Host', 'UniqueID', 'Status', 'Command+Args'))
        for config_id, detail in runner.config.items():
            comment = sbman.get_comment(runner.xpm_id, runner.platform, config_id)
            if comment in job_details:
                job_detail = job_details[comment]
                print('%20s %12s %10s %40s'%(job_detail['nodelist'], job_detail['job_name'], job_detail['state'], detail['cmd']))

def cancel(slurm_job_id):
    if runner is None: return
    sbatch_cmd = f"scancel {slurm_job_id}"
    asyncio.run(proc.run(sbatch_cmd, wait_output=True))

def stop():
    """ Stops running job using their comment.
    
    Each job is submitted with their unique comment. We can stop all the processes
    by looking at the given cnf and match the comment (see below for detail) with
    comment returned by slurm."""
    if runner is None: return
    job_comments = {}
    for i, job_info in enumerate(sbman.get_job_info(format_string='"%i %k"', noheader=True)):
        job_id, comment = job_info.strip('"').split()
        job_comments[comment] = job_id

    for config_id, detail in runner.config.items():
        comment = sbman.get_comment(runner.xpm_id, runner.platform, config_id)
        if comment in job_comments:
            cancel(job_comments[comment])

def restart():
    stop()
    start()
    
def _do_main():
    typer.run(main)

if __name__ == "__main__":
    start()
