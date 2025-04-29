import os

platform = '0'

CONDA_PREFIX = os.environ.get('CONDA_PREFIX','')
#CONFIGDIR = '/cds/home/m/monarin/lcls2/psdaq/psdaq/slurm'
host, cores, id, flags, env, cmd, rtprio = ('host', 'cores', 'id', 'flags', 'env', 'cmd', 'rtprio')

ld_lib_path = f'LD_LIBRARY_PATH={CONDA_PREFIX}/epics/lib/linux-x86_64:{CONDA_PREFIX}/pcas/lib/linux-x86_64'
epics_env = ld_lib_path

xpm_nogateway_epics_env = f'EPICS_PVA_ADDR_LIST=172.21.152.78 EPICS_PVA_AUTO_ADDR_LIST=NO {ld_lib_path}'
feespec_epics_env = f'EPICS_CA_ADDR_LIST=172.21.88.87 EPICS_CA_AUTO_ADDR_LIST=NO {ld_lib_path}'
alvium_epics_env = f'EPICS_CA_ADDR_LIST=172.21.72.161 EPICS_CA_AUTO_ADDR_LIST=NO {ld_lib_path}'

collect_host = 'drp-srcf-cmp014'

groups = f'{platform} 1 5'
hutch, station, user = ('mfx', 1, 'mfxopr')
hutch_meb = 'mfx'
auth = ' --user {:} '.format(user)
url  = ' --url https://pswww.slac.stanford.edu/ws-auth/lgbk/ '
cdb  = 'https://pswww.slac.stanford.edu/ws-auth/configdb/ws'

#
#  drp variables
#
prom_dir = f'/cds/group/psdm/psdatmgr/etc/config/prom/{hutch}' # Prometheus
data_dir = '/cds/data/drpsrcf'
trig_dir = '/cds/group/pcds/dist/pds/mfx/scripts/trigger'

task_set = 'taskset -c 4-63'
scripts  = f'script_path={trig_dir}'

std_opts = f'-P {hutch} -C {collect_host} -M {prom_dir}'
std_opts0 = f'{std_opts} -o {data_dir} -d /dev/datadev_0'
std_opts1 = f'{std_opts} -o {data_dir} -d /dev/datadev_1'

teb_cmd  = f'{task_set} teb {std_opts} -k {scripts}'
meb_cmd  = f'{task_set} monReqServer {std_opts}'

drp_cmd0 = f'{task_set} drp {std_opts0}'
drp_cmd1 = f'{task_set} drp {std_opts1}'
pva_cmd0 = f'{task_set} drp_pva {std_opts0}'
pva_cmd1 = f'{task_set} drp_pva {std_opts1}'

elog_cfg = '/cds/group/pcds/dist/pds/mfx/misc/logbook.txt'
ea_cfg = '/cds/group/pcds/dist/pds/mfx/misc/epicsArch.txt'

ea_cmd0 = f'{task_set} epicsArch {std_opts0} {ea_cfg}'
ea_cmd1 = f'{task_set} epicsArch {std_opts1} {ea_cfg}'

# (8,512,1024)
jungfrau_img_path = "/cds/home/d/dorlhiac/xpptut15_run580_jungfrau_img1.bin"
jungfrauemu_kwargs = f"sim_length=262144,imgArray={jungfrau_img_path}"

segnums_0 = "segNums=0_1_2_3"
segnums_1 = "segNums=7_8_9_10"

jungfrauemu_kwargs_0 = f"{jungfrauemu_kwargs},{segnums_0}"
jungfrauemu_kwargs_1 = f"{jungfrauemu_kwargs},{segnums_1}"


#
#  ami variables
#
heartbeat_period = 1000 # units are ms

ami_workers_per_node = 4
ami_worker_nodes = ["drp-srcf-mon003"]
ami_num_workers = len(ami_worker_nodes)
ami_manager_node = "drp-srcf-mon003"

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
 #{                                     id: 'xpmpva' ,     flags: 's',   env: xpm_nogateway_epics_env, cmd: 'xpmpva DAQ:FEH:XPM:2'},
 {                                     id: 'xpmpva' ,     flags: 's',   env: epics_env,               cmd: 'xpmpva DAQ:FEH:XPM:2 DAQ:FEH:XPM:3'},
 {                                     id: 'groupca',     flags: 's',   env: epics_env, cmd: f'groupca DAQ:FEH 2 {groups}'},
 {                                     id: 'daqstat',                                                 cmd: 'daqstat -i 5'},

 { host: collect_host,                 id: 'control',     flags: 'spu', env: epics_env,               cmd: f'control -P {hutch}:{station} -B DAQ:FEH -x 2 -C BEAM {auth} {url} -d {cdb}/configDB -t trigger -S 1 -T 20000 -V {elog_cfg} -c'},
 {                                     id: 'control_gui', flags: 'p',                                 cmd: f'control_gui -H {collect_host} --uris {cdb} --expert {auth} --loglevel WARNING'},

 { host: 'drp-srcf-cmp014',            id: 'teb0',        flags: 'spu',                               cmd: teb_cmd},
 { host: 'drp-srcf-cmp014',            id: 'mebuser0',    flags: 'spu',                               cmd: f'{meb_cmd} -t {hutch_meb}_mebuser0 -d -n 64 -q {ami_workers_per_node}'},
 { host: 'drp-srcf-cmp014', cores: 10, id: 'timing_1',    flags: 'spu', env: epics_env,               cmd: f'{drp_cmd1} -l 0x1 -D ts'},

 # Jungfrau emulator
 { host: 'drp-srcf-cmp033', cores: 10, id: 'jungfrau_0',  flags: 'spu', env: epics_env,               cmd: f'{drp_cmd0} -D jungfrauemu -l 0xf -k {jungfrauemu_kwargs_0}'},

 #{ host: 'drp-srcf-cmp034', cores: 10, id: 'jungfrau_1',  flags: 'spu', env: epics_env,               cmd: f'{drp_cmd0} -D jungfrauemu -l 0xf -k {jungfrauemu_kwargs_1}'},

 { host: 'drp-srcf-cmp035', cores: 10, id: 'jungfrau_1',  flags: 'spu', env: epics_env,               cmd: f'{drp_cmd0} -D jungfrauemu -l 0xf -k {jungfrauemu_kwargs_1}'},
 #{ host: 'drp-srcf-cmp035', cores: 10, id: 'jungfrau_2',  flags: 'spu', env: epics_env,               cmd: f'{drp_cmd0} -D jungfrauemu -l 0xf -k {jungfrauemu_kwargs}'},
]

# cpo: this formula should be base_port=5555+50*platform+10*instance
# to minimize port conflicts for both platform/instance
platform_base_port = 5555+50*int(platform)
for instance, base_port in {"first": platform_base_port,}.items():
    procmgr_ami = [
      { host:ami_manager_node, id:f'ami-global_{instance}',  flags:'s', env:epics_env, cmd:f'ami-global -p {base_port} --hutch {hutch_meb}_{instance} --prometheus-dir {prom_dir} -N 0 -n {ami_num_workers}' },
      { host:ami_manager_node, id:f'ami-manager_{instance}', flags:'s', cmd:f'ami-manager -p {base_port} --hutch {hutch_meb}_{instance} --prometheus-dir {prom_dir} -n {ami_num_workers*ami_workers_per_node} -N {ami_num_workers}' },
      {                        id:f'ami-client_{instance}',  flags:'s', cmd:f'ami-client -p {base_port} -H {ami_manager_node} --prometheus-dir {prom_dir} --hutch {hutch_meb}_{instance}' },

    ]

    # ami workers
    for N, worker_node in enumerate(ami_worker_nodes):
        procmgr_ami.append({ host:worker_node, id:f'meb{N}_{instance}', flags:'spu',
                             cmd:f'{meb_cmd} -t {hutch_meb}{instance} -d -n 64 -q {ami_workers_per_node}' })
        procmgr_ami.append({ host:worker_node, id:f'ami-node_{N}_{instance}', flags:'s', env:epics_env,
                             cmd:f'ami-node -p {base_port} --hutch {hutch_meb}_{instance} --prometheus-dir {prom_dir} -N {N} -n {ami_workers_per_node} -H {ami_manager_node} --log-level info worker -b {heartbeat_period} psana://shmem={hutch_meb}{instance}' })

    procmgr_config.extend(procmgr_ami)
