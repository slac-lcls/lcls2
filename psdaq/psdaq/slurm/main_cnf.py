platform = "4"

import os

CONDA_PREFIX = os.environ.get("CONDA_PREFIX", "")
CONFIGDIR = "/cds/home/m/monarin/lcls2/psdaq/psdaq/slurm"
host, cores, id, flags, env, cmd, rtprio = (
    "host",
    "cores",
    "id",
    "flags",
    "env",
    "cmd",
    "rtprio",
)
task_set = ""

ld_lib_path = f"LD_LIBRARY_PATH={CONDA_PREFIX}/epics/lib/linux-x86_64:{CONDA_PREFIX}/pcas/lib/linux-x86_64"
epics_env = f"{ld_lib_path}"

collect_host = "drp-srcf-cmp035"

groups = platform + " 1"
hutch, user = ("tst", "tstopr")
auth = " --user {:} ".format(user)
# url  = ' --url https://pswww.slac.stanford.edu/ws-auth/lgbk/ '
url = " --url https://pswww.slac.stanford.edu/ws-auth/devlgbk/ "
cdb = "https://pswww.slac.stanford.edu/ws-auth/configdb/ws"

#
#  drp variables
#
prom_dir = f"/cds/group/psdm/psdatmgr/etc/config/prom/{hutch}"  # Prometheus
data_dir = f"/cds/data/drpsrcf"
trig_dir = f"/cds/home/c/claus/lclsii/daq/runs/eb/data/srcf"

# task_set = 'taskset -c 4-63'
task_set = ""
batching = "batching=yes"
directIO = "directIO=yes"
scripts = f"script_path={trig_dir}"
# network  = 'ep_provider=sockets,ep_domain=eno1'

std_opts = f"-P {hutch} -C {collect_host} -M {prom_dir}"  # -k {network}'
std_opts0 = f"{std_opts} -d /dev/datadev_0 -o {data_dir} -k {batching},{directIO}"
std_opts1 = f"{std_opts} -d /dev/datadev_1 -o {data_dir} -k {batching},{directIO}"

teb_cmd = f"{task_set} teb " + std_opts + f" -k {scripts}"
meb_cmd = f"{task_set} monReqServer {std_opts}"

drp_cmd0 = f"{task_set} drp " + std_opts0
drp_cmd1 = f"{task_set} drp " + std_opts1
pva_cmd0 = f"{task_set} drp_pva {std_opts0}"
pva_cmd1 = f"{task_set} drp_pva {std_opts1}"
bld_cmd0 = f"{task_set} drp_bld {std_opts0} -k interface=eno1"
bld_cmd1 = f"{task_set} drp_bld {std_opts1} -k interface=eno1"

elog_cfg = f"/cds/group/pcds/dist/pds/{hutch}/misc/elog_{hutch}.txt"

# ea_cfg = f'/cds/group/pcds/dist/pds/{hutch}/misc/epicsArch.txt'
# ea_cfg = f'/cds/group/pcds/dist/pds/tmo/misc/epicsArch_tmo.txt'
ea_cfg = f"/cds/group/pcds/dist/pds/rix/misc/epicsArch.txt"
ea_cmd0 = f"{task_set} epicsArch {std_opts0} {ea_cfg}"
ea_cmd1 = f"{task_set} epicsArch {std_opts1} {ea_cfg}"

#
#  ami variables
#
heartbeat_period = 1000  # units are ms

ami_workers_per_node = 4
ami_worker_nodes = ["drp-srcf-cmp035"]
ami_num_workers = len(ami_worker_nodes)
ami_manager_node = "drp-srcf-cmp035"
ami_monitor_node = "drp-srcf-cmp035"

# procmgr FLAGS: <port number> static port number to keep executable
#                              running across multiple start/stop commands.
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
    {
        id: "psqueue",
        cmd: f"psqueue -i 5",
    },
    # set the phase2 transition timeout to 20s. this is because the teb
    # has a 16s timeout for slow PVA detectors coming through the gateway.
    # both can be reduced if/when we get consistent behavior from gateways.
    {
        host: collect_host,
        id: "control",
        flags: "spux",
        env: epics_env,
        cmd: f"control -P {hutch} -B DAQ:NEH -x 0 -C BEAM {auth} {url} -d {cdb}/configDB -t trigger -S 1 -T 20000 -V {elog_cfg}",
    },
    {
        id: "control_gui",
        flags: "p",
        cmd: f"control_gui -H {collect_host} --uris {cdb} --expert {auth} --loglevel WARNING",
    },
    {host: "drp-srcf-cmp035", id: "teb0", flags: "spu", cmd: f"{teb_cmd}"},
    {
        host: "drp-srcf-cmp002",
        id: "timing_0",
        flags: "spu",
        env: epics_env,
        cmd: f"{drp_cmd1} -l 0x1 -D ts",
    },
]

#
# ami
#
ami_config = [
    {
        host: ami_manager_node,
        id: "ami-global",
        flags: "s",
        env: epics_env,
        cmd: f"ami-global --hutch {hutch} --prometheus-dir {prom_dir} -N 0 -n {ami_num_workers}",
    },
    {
        host: ami_manager_node,
        id: "ami-manager",
        flags: "s",
        cmd: f"ami-manager --hutch {hutch} --prometheus-dir {prom_dir} -n {ami_num_workers*ami_workers_per_node} -N {ami_num_workers}",
    },
    {
        id: "ami-client",
        flags: "s",
        cmd: f"ami-client -H {ami_manager_node} --prometheus-dir {prom_dir} --hutch {hutch}",
    },
]

#
# ami workers
#
for N, worker_node in enumerate(ami_worker_nodes):
    ami_config.append(
        {
            host: worker_node,
            id: f"ami-meb{N}",
            flags: "spu",
            cmd: f"{meb_cmd} -d -n 64 -q {ami_workers_per_node}",
        }
    )
    ami_config.append(
        {
            host: worker_node,
            id: f"ami-node_{N}",
            flags: "s",
            env: epics_env,
            cmd: f"ami-node --hutch {hutch} --prometheus-dir {prom_dir} -N {N} -n {ami_workers_per_node} -H {ami_manager_node} --log-level warning worker -b {heartbeat_period} psana://shmem={hutch}",
        }
    )

# procmgr_config.extend(ami_config)
