platform = '0'

import os
CONDA_PREFIX = os.environ.get('CONDA_PREFIX','')
CONFIGDIR = '/cds/home/m/monarin/lcls2/psdaq/psdaq/slurm'
host, cores, id, flags, env, cmd, rtprio = ('host', 'cores', 'id', 'flags', 'env', 'cmd', 'rtprio')
task_set = ''


ld_lib_path = f'LD_LIBRARY_PATH={CONDA_PREFIX}/epics/lib/linux-x86_64:{CONDA_PREFIX}/pcas/lib/linux-x86_64 PYEPICS_LIBCA={CONDA_PREFIX}/epics/lib/linux-x86_64/libca.so'

epics_env = ld_lib_path
xpm_nogateway_epics_env = 'EPICS_PVA_ADDR_LIST=172.21.152.78 EPICS_PVA_AUTO_ADDR_LIST=NO'+' '+ld_lib_path
manta_env = "EPICS_PVA_ADDR_LIST=172.21.140.33 EPICS_PVA_AUTO_ADDR_LIST=NO"+' '+ld_lib_path
mr2k1_env = "EPICS_PVA_ADDR_LIST=172.21.92.68 EPICS_PVA_AUTO_ADDR_LIST=NO"+' '+ld_lib_path
hsd_epics_env = 'EPICS_PVA_ADDR_LIST=172.21.152.255'+' '+ld_lib_path
andor_epics_env = 'EPICS_PVA_ADDR_LIST=172.21.140.55 EPICS_PVA_AUTO_ADDR_LIST=YES'+' '+ld_lib_path
archon_epics_env = 'EPICS_PVA_ADDR_LIST=daq-rix-ccd-01 EPICS_PVA_AUTO_ADDR_LIST=YES'+' '+ld_lib_path

collect_host = 'drp-srcf-cmp004'

# readout group 4 is for epixhr (L0Delay change)
# readout group 5 is for manta
groups = f'1 {platform} 5'
#groups = f'1 {platform} 5 7'
hutch, station, user = ('rix', 2, 'rixopr')
#hutch, station, user = ('tst', 2, 'tstopr')
auth = ' --user {:} '.format(user)
url  = ' --url https://pswww.slac.stanford.edu/ws-auth/lgbk/ '
cdb  = 'https://pswww.slac.stanford.edu/ws-auth/configdb/ws'

#
#  drp variables
#
prom_dir = f'/cds/group/psdm/psdatmgr/etc/config/prom/{hutch}' # Prometheus
data_dir = '/cds/data/drpsrcf'
trig_dir = '/cds/home/opr/rixopr/scripts/trigger'

pvaAddr  = 'pva_addr=172.21.152.78' # mon001
scripts  = f'script_path={trig_dir}'
network  = 'ep_provider=sockets,ep_domain=eno1'
encKwargs = 'pebbleBufCount=1048576,pebbleBufSize=4096,encTprAlias=tprtrig'

std_opts = f'-P {hutch} -C {collect_host} -M {prom_dir}' # -k {network}'
std_opts0 = f'{std_opts} -o {data_dir} -d /dev/datadev_0 -k {pvaAddr}'
std_opts1 = f'{std_opts} -o {data_dir} -d /dev/datadev_1 -k {pvaAddr}'

teb_cmd  = task_set+' teb '            +std_opts+' -k '+scripts
meb_cmd  = task_set+' monReqServer '   +std_opts

drp_cmd0 = task_set+' drp '            +std_opts0
drp_cmd1 = task_set+' drp '            +std_opts1
pva_cmd0 = task_set+' drp_pva '        +std_opts0
pva_cmd1 = task_set+' drp_pva '        +std_opts1
udp_cmd0 = task_set+' drp_udpencoder ' +std_opts0+f' -k {encKwargs}'
udp_cmd1 = task_set+' drp_udpencoder ' +std_opts1+f' -k {encKwargs}'
tpr_cmd  = task_set+' tpr_trigger '    +f' -H {hutch} -C {collect_host}'
bld_cmd0 = task_set+' drp_bld '        +std_opts0+' -k interface=eno1 -k pva_addr=172.27.131.255'
bld_cmd1 = task_set+' drp_bld '        +std_opts1+' -k interface=eno1 -k pva_addr=172.27.131.255'

#  format is {name}+{type}+{pvname}
gmd  = 'gmd+scbld+EM1K0:GMD:HPS'
xgmd = 'xgmd+scbld+EM2K0:XGMD:HPS'

elog_cfg = '/cds/group/pcds/dist/pds/rix/misc/elog_rix.txt'

ea_cfg = '/cds/group/pcds/dist/pds/rix/misc/epicsArch.txt'
ea_cmd0 = task_set+' epicsArch '+std_opts0+' '+ea_cfg
ea_cmd1 = task_set+' epicsArch '+std_opts1+' '+ea_cfg

#
#  ami variables
#
heartbeat_period = 1000 # units are ms

ami_workers_per_node = 4
ami_worker_nodes = ["drp-srcf-cmp029"]
ami_num_workers = len(ami_worker_nodes)
ami_manager_node = "drp-srcf-cmp029"

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
 {                          id:'xpmpva' ,     flags:'s',   env:epics_env, cmd:'xpmpva DAQ:NEH:XPM:3 DAQ:NEH:XPM:5'},
 {                          id:'groupca',     flags:'s',   env:epics_env, cmd:'groupca DAQ:NEH 3 '+groups},
 {id: 'daqstat', cmd: 'daqstat -i 5'},
 { host: collect_host,      id:'control',     flags:'spu', env:xpm_nogateway_epics_env, cmd:f'control -P {hutch}:{station} -B DAQ:NEH -x 3 -X drp-srcf-mon001 -C BEAM {auth} {url} -d {cdb}/configDB -t trigger -S 1 -T 20000 -V {elog_cfg} -c'},
 {                          id:'control_gui', flags:'p',                  cmd:f'control_gui -H {collect_host} --uris {cdb} --expert {auth} --loglevel WARNING'},

 #{ host: 'drp-neh-ctl002',  id:'tprtrig',     flags:'sp',                 cmd:'tprtrig -t a -c 0 -o 1 -d 2 -w 10 -z'},
 { host: 'drp-neh-ctl002', id:'mono_enc_trig',     flags:'spu',                 cmd:f'{tpr_cmd} -t a -c 0 -o 1 -d 2 -w 10'},

 { host: 'drp-srcf-cmp013', id:'teb0',        flags:'spu',                cmd:teb_cmd},

 { host: 'drp-srcf-cmp008', id:'mebuser0',    flags:'spu',                cmd:meb_cmd+f' -t {hutch}_mebuser0 -d -n 64 -q {ami_workers_per_node}'},

 { host: 'drp-srcf-cmp002', cores: 10, id:'timing_0',    flags:'spu', env:xpm_nogateway_epics_env, cmd:drp_cmd1+' -l 0x1 -D ts'},

 { host: 'drp-srcf-cmp002', id:'bld_0',       flags:'spu', env:epics_env, cmd:bld_cmd1+' -l 0x2 -D '+f'{gmd},{xgmd}'},

 { host: 'drp-srcf-cmp010', id:'andor_dir_0', flags:'spu', env:andor_epics_env, cmd:pva_cmd1+' -l 0x4 RIX:DIR:CAM:01:IMAGE1:Pva:Image -k pebbleBufSize=2098216'},

 { host: 'drp-srcf-cmp010', id:'andor_norm_0', flags:'spu', env:andor_epics_env, cmd:pva_cmd0+' -l 0x2 RIX:NORM:CAM:01:IMAGE1:Pva:Image -k pebbleBufSize=2098216'},

 { host: 'drp-srcf-cmp010', id:'andor_vls_0', flags:'spu', env:andor_epics_env, cmd:pva_cmd0+' -l 0x1 RIX:VLS:CAM:01:IMAGE1:Pva:Image -k pebbleBufSize=2098216'},

 { host: 'drp-srcf-cmp010', id:'archon_0', flags:'spu', env:archon_epics_env, cmd:pva_cmd1+' -D archon -S 1 -l 0x2 QRIX:STA:CCD:01:IMAGE1:Pva:Image -k pebbleBufSize=11521064'},

 # make transition buffer size large to accomodate a non-timestamped camera
 { host: 'drp-srcf-cmp010', id:'epics_0',     flags:'spu', env:epics_env,  cmd:ea_cmd1+' -l 0x8 -T 4000000'},

 { host: 'drp-srcf-cmp002', id:'mono_encoder_0',   flags:'spu', rtprio:'50', cmd:udp_cmd0+' -l 0x8'},
 { host: 'drp-srcf-cmp004',cores:10, id:'mono_hrencoder_0', flags:'spu',              cmd:f'{drp_cmd0} -l 0x80 -D hrencoder'},

# This node is shared with TMO, so need to take care not to over-allocate lanes
 { host: 'drp-srcf-cmp004',cores: 10, id:'crix_w8_0',  flags:'spu', env:epics_env,  cmd:drp_cmd0+' -l 0x1 -D wave8 -k epics_prefix=RIX:CRIX:W8:01'},
 { host: 'drp-srcf-cmp004',cores: 10, id:'rix_fim1_0',  flags:'spu', env:epics_env,  cmd:drp_cmd0+' -l 0x20 -D wave8 -k epics_prefix=MR4K2:FIM:W8:02'},
 { host: 'drp-srcf-cmp004',cores: 10, id:'rix_fim0_0',  flags:'spu', env:epics_env,  cmd:drp_cmd0+' -l 0x4 -D wave8 -k epics_prefix=MR3K2:FIM:W8:01'},

  #{ host: 'drp-srcf-cmp027',cores: 10, id:'atmopal_0',   flags:'spu', env:epics_env,  cmd:drp_cmd0+' -l 0x1 -D opal -k ttpv=RIX:TIMETOOL:TTALL'},
  { host: 'drp-srcf-cmp027',cores: 10, id:'c_atmopal_0',   flags:'spu', env:epics_env,  cmd:drp_cmd0+' -l 0x1 -D opal'},
  { host: 'drp-srcf-cmp027',cores: 10, id:'q_atmopal_0',   flags:'spu', env:epics_env,  cmd:drp_cmd0+' -l 0x2 -D opal'},
 #{ host: 'drp-srcf-cmp012',cores: 10, id:'c_piranha_0',   flags:'spu', env:epics_env,  cmd:drp_cmd0+' -l 0x2 -D piranha4'},
 { host: 'drp-srcf-cmp012',cores: 10, id:'c_piranha_0',   flags:'spu', env:epics_env,  cmd:drp_cmd0+' -l 0x2 -D piranha4 -k ttpv=RIX:TIMETOOL:TTALL'},
 { host: 'drp-srcf-cmp012',cores: 10, id:'q_piranha_0',   flags:'spu', env:epics_env,  cmd:drp_cmd0+' -l 0x4 -D piranha4'},

# Temporarily borrow cmp012 from TMO (Opal1) to allow RIX atmopal and piranha to run at the same time
# { host: 'drp-srcf-cmp012',cores: 10, id:'piranha_0',   flags:'spu', env:epics_env, cmd:drp_cmd0+' -l 0x1 -D piranha4'},

 #     Note: '-1' specifies fuzzy timestamping
 #           '-0' specifies no timestamp matching (-0 may have problems)
 #           ''   specifies precise timestamp matching
 # pebbleBufSize was 8388672
 { host: 'drp-srcf-cmp025', id:'manta_0',     flags:'spu', env:manta_env, cmd:pva_cmd1+' -l 0x1 SL1K2:EXIT:CAM:DATA1:Pva:Image -k pebbleBufSize=8400000'},
 # 9MPixel G-917C manta untimestamped image @1Hz (hence the "-0" flag)
 # set the pebbleBufSize smaller since filewriter only supports 8388688 currently
 { host: 'drp-srcf-cmp010', id:'mr2k1_0',     flags:'spu', env:mr2k1_env, cmd:pva_cmd1+' -l 0x1 MR2K1:MONO:CAM:04:DATA1:Pva:Image -k pebbleBufSize=8380000 -0'},
]

hsd_epics = 'DAQ:RIX:HSD:1'

procmgr_hsd = [
 {host:'drp-srcf-cmp009',cores: 10, id:'hsd_0',  flags:'spu', env:hsd_epics_env, cmd:drp_cmd1+f' -D hsd -k hsd_epics_prefix={hsd_epics}_1A:A'},
 {host:'drp-srcf-cmp009',cores: 10, id:'hsd_1',  flags:'spu', env:hsd_epics_env, cmd:drp_cmd0+f' -D hsd -k hsd_epics_prefix={hsd_epics}_1A:B'},
 {host:'drp-srcf-cmp005',cores: 10, id:'hsd_2',  flags:'spu', env:hsd_epics_env, cmd:drp_cmd0+f' -D hsd -k hsd_epics_prefix={hsd_epics}_1B:B'},
 {host:'drp-srcf-cmp005',cores: 10, id:'hsd_3',  flags:'spu', env:hsd_epics_env, cmd:drp_cmd1+f' -D hsd -k hsd_epics_prefix={hsd_epics}_1B:A'},
]

procmgr_config.extend(procmgr_hsd)

# cpo: this formula should be base_port=5555+50*platform+10*instance
# to minimize port conflicts for both platform/instance
platform_base_port = 5555+50*int(platform)
for instance, base_port in {"first": platform_base_port,}.items():
    procmgr_ami = [
      { host:ami_manager_node, id:f'ami-global_{instance}',  flags:'s', env:epics_env, cmd:f'ami-global -p {base_port} --hutch {hutch}_{instance} --prometheus-dir {prom_dir} -N 0 -n {ami_num_workers}' },
      { host:ami_manager_node, id:f'ami-manager_{instance}', flags:'s', cmd:f'ami-manager -p {base_port} --hutch {hutch}_{instance} --prometheus-dir {prom_dir} -n {ami_num_workers*ami_workers_per_node} -N {ami_num_workers}' },
      {                        id:f'ami-client_{instance}',  flags:'s', cmd:f'ami-client -p {base_port} -H {ami_manager_node} --prometheus-dir {prom_dir} --hutch {hutch}_{instance}' },
#      {                        id:f'ami-client_{instance}_acr',  flags:'s', cmd:f'ami-client -l rix_acr_hsd_feedback.fc -p {base_port} -H {ami_manager_node} --prometheus-dir {prom_dir} --hutch {hutch}_{instance}_acr -g "acr_graph"'},
#      {                        id:f'acr_server',  flags:'s', cmd:f'python rix_acr_hsd_feedback.py -P RIX:ACR'},
    ]

    # ami workers
    for N, worker_node in enumerate(ami_worker_nodes):
        procmgr_ami.append({ host:worker_node, id:f'meb{N}_{instance}', flags:'spu',
                             cmd:f'{meb_cmd} -t {hutch}{instance} -d -n 64 -q {ami_workers_per_node}' })
        procmgr_ami.append({ host:worker_node, id:f'ami-node_{N}_{instance}', flags:'s', env:epics_env,
                             cmd:f'ami-node -p {base_port} --hutch {hutch}_{instance} --prometheus-dir {prom_dir} -N {N} -n {ami_workers_per_node} -H {ami_manager_node} --log-level warning worker -b {heartbeat_period} psana://shmem={hutch}{instance}' })

    procmgr_config.extend(procmgr_ami)
