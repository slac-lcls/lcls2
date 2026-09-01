platform = '0'

import os
CONDA_PREFIX = os.environ.get('CONDA_PREFIX','')
CONFIGDIR = '/cds/home/m/monarin/lcls2/psdaq/psdaq/slurm'
host, cores, id, flags, env, env_group, cmd, rtprio = ('host', 'cores', 'id', 'flags', 'env', 'env_group', 'cmd', 'rtprio')
task_set = ''


ld_lib_path = f'LD_LIBRARY_PATH={CONDA_PREFIX}/epics/lib/linux-x86_64:{CONDA_PREFIX}/pcas/lib/linux-x86_64'
epics_env = 'EPICS_PVA_ADDR_LIST=172.21.156.255'+' '+ld_lib_path

collect_host = 'drp-neh-ctl002'

groups = platform
hutch, user, password = ('tst', 'tstopr', 'pcds')
auth = f' --user {user} --password {password} '

cdb  = 'https://pswww.slac.stanford.edu/ws-auth/configdb/ws'

#
#  drp variables
#
prom_dir = '/cds/group/psdm/psdatmgr/etc/config/prom' # Prometheus
data_dir = '/reg/neh/operator/tstopr/data/drp'
trig_dir = '/cds/home/m/monarin/lcls2/install/lib/python3.9/site-packages/psdaq/trigger'
scripts  = f'script_path={trig_dir}'

std_opts =f'-P {hutch} -C {collect_host} -M {prom_dir} -o {data_dir} -d /dev/datadev_%d'

drp_cmd0 = task_set+'drp '      +(std_opts%0)+' -k batching=no '
drp_cmd1 = task_set+'drp '      +(std_opts%1)+' -k batching=no '
pva_cmd0 = task_set+'drp_pva '  +(std_opts%0)+' -k batching=no '
pva_cmd1 = task_set+'drp_pva '  +(std_opts%1)+' -k batching=no '
bld_cmd0 = task_set+'drp_bld '  +(std_opts%0)+' -k batching=no,interface=eno1 '
bld_cmd1 = task_set+'drp_bld '  +(std_opts%1)+' -k batching=no,interface=eno1 '
teb_cmd  = task_set+f'teb -P {hutch} -C {collect_host} -M {prom_dir} -k {scripts}'

ea_cfg = '/cds/group/pcds/dist/pds/tmo/misc/epicsArch_fee.txt'
ea_cmd0 = task_set+'epicsArch '+(std_opts%1)+' -k batching=no '+ea_cfg
ea_cmd1 = task_set+'epicsArch '+(std_opts%1)+' -k batching=no '+ea_cfg

#
#  ami variables
#
heartbeat_period = 1000 # units are ms

ami_workers_per_node = 4
ami_worker_nodes = ["drp-neh-cmp013"]
ami_num_workers = len(ami_worker_nodes)
ami_manager_node = "drp-neh-cmp016"

# procmgr FLAGS: <port number> static port number to keep executable
#                              running across multiple start/stop commands.
#                "X" open xterm
#                "s" send signal to child when stopping
#
# HOST       UNIQUEID      FLAGS  COMMAND+ARGS
# list of processes to run
#   required fields: id, cmd
#   optional fields: host, port, flags, conda, env, rtprio
#     flags:
#        'x' or 'X'  -> xterm: open small or large xterm for process console
#        's'         -> stop: sends ctrl-c to process
#        'u'         -> uniqueid: use 'id' as detector alias (supported by acq, cam, camedt, evr, and simcam)

procmgr_config = [
 {                         id:'xpmpva' ,     flags:'s',   env:epics_env, cmd:'xpmpva DAQ:NEH:XPM:10'},
 {                         id:'groupca',     flags:'s',   env:epics_env, cmd:f'groupca DAQ:NEH 10 {groups}'},
 {id: 'daqstat', cmd: 'daqstat -i 5'},
 { host: collect_host,     id:'control',     flags:'spu', env:epics_env, cmd:f'control -P {hutch} -B DAQ:NEH -x 10 -C BEAM {auth} -d {cdb}/configDB -t trigger -S 1 -T 20000'},
 {                         id:'control_gui', flags:'p',                  cmd:f'control_gui -H {collect_host} --uris {cdb} --expert {auth} --loglevel WARNING'},

 { host: 'drp-neh-cmp016', id:'teb0',        flags:'spu',                cmd:teb_cmd},

 { host: 'drp-neh-cmp002', id:'timing_0',    flags:'spux',                cmd:f'{drp_cmd1} -l 0x1 -D ts'},
 #{ host: 'drp-neh-cmp002', id:'timing_0',    flags:'spu',                cmd:f'{drp_cmd1} -l 0x1 -D ts -k drp=python -k pythonScript=/cds/home/m/monarin/psana-nersc/psana2/drp_python/test_drp.py'},
 #{ host: 'drp-neh-cmp015', id:'timing_1',    flags:'spux',                cmd:f'{drp_cmd1} -l 0x2 -D ts'},
 #{ host: 'drp-neh-cmp002', id:'timing_1',    flags:'spux',                cmd:f'{drp_cmd1} -l 0x2 -D ts -k drp=python -k pythonScript=/cds/home/m/monarin/psana-nersc/psana2/drp_python/test_drp.py'},
 #{ host: 'drp-neh-cmp015', id:'timing_4',    flags:'spux',                cmd:f'{drp_cmd1} -l 0x1 -D fakecam'}

]

procmgr_ami = [
#
# ami
#
 { host:ami_manager_node, id:'ami-global',  flags:'s', env_group: 'ami', cmd:f'ami-global --hutch {hutch} --prometheus-dir {prom_dir} -N 0 -n {ami_num_workers}' },
 { host:ami_manager_node, id:'ami-manager', flags:'s', env_group: 'ami', cmd:f'ami-manager --hutch {hutch} --prometheus-dir {prom_dir} -n {ami_num_workers*ami_workers_per_node} -N {ami_num_workers}' },
 {                        id:'ami-client',  flags:'s', env_group: 'ami', cmd:f'ami-client -H {ami_manager_node} --prometheus-dir {prom_dir}/{hutch}' },
]

#
# ami workers
#
for N, worker_node in enumerate(ami_worker_nodes):
    procmgr_ami.append({ host:worker_node, id:f'meb{N}', flags:'spu', env_group: 'ami',
                         cmd:f'{task_set} monReqServer -P {hutch} -C {collect_host} -M {prom_dir} -d -q {ami_workers_per_node}' })
    procmgr_ami.append({ host:worker_node, id:f'ami-node_{N}', flags:'s', env_group: 'ami',
                         cmd:f'ami-node --hutch {hutch} --prometheus-dir {prom_dir} -N {N} -n {ami_workers_per_node} -H {ami_manager_node} --log-level warning worker -b {heartbeat_period} psana://shmem={hutch}' })

# Disable AMI 
procmgr_config.extend(procmgr_ami)
