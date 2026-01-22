import os

if not platform: platform = '7'

CONDA_PREFIX = os.environ.get("CONDA_PREFIX","")
host, cores, id, flags, env, cmd, rtprio, port = ("host", "cores", "id", "flags", "env", "cmd", "rtprio", "port")
ld_lib_path = f"LD_LIBRARY_PATH={CONDA_PREFIX}/epics/lib/linux-x86_64:{CONDA_PREFIX}/pcas/lib/linux-x86_64"
epics_env = f"EPICS_PVA_ADDR_LIST=172.21.159.255 {ld_lib_path}"

def pyxpm_optsdb(n,ext='NEH'):
    #  Xilinx Virtual Cable connections
    def xvc_arg(node):
        xvc_ports = {'pyxpm-0'     : 2542,   # drp-srcf-mon001
                     'pyxpm-1'     : 2543,
                     'pyxpm-2'     : 2544,
                     'pyxpm-3'     : 2545,
                     'pyxpm-4'     : 2546,
                     'pyxpm-5'     : 2547,
                     'pyxpm-6'     : 2548,
                     'pyxpm-feh-0' : 2549,
                     'pyxpm-feh-1' : 2550,
                     'pyxpm-feh-2' : 2551,
                     'pyxpm-feh-3' : 2542,   # drp-srcf-cmp043
                     'pyxpm-10'    : 2542,   # drp-neh-ctl002
                     'pyxpm-11'    : 2543,
                     'pyxpm-12'    : 2544,
		     'pyxpm-feh-4' : 2552,
		     'pyxpm-feh-5' : 2553,
		     'pyxpm-feh-6' : 2554 
        }
        xvc_ports = {}
        if not node in xvc_ports:
            return ' '
        else:
            return f' --xvc {xvc_ports[node]}'

    node = f'pyxpm-feh-{n}' if ext=='FEH' else f'pyxpm-{n}'
    return f"--db https://pswww.slac.stanford.edu/ws-auth/configdb/ws/,configDB,tmo,XPM -P DAQ:{ext}:XPM:{n} {xvc_arg(node)}"

kcu_host = "drp-srcf-cmp043"
xpp_kcu_host = "drp-srcf-mon009"
gpu_kcu_host = "drp-srcf-mon010"
base_host = "drp-srcf-mon001"
feh_host = "drp-srcf-mon001"
fee_host  = "drp-neh-ctl002"
prom_dir = "/cds/group/psdm/psdatmgr/etc/config/prom/base" # Prometheus

procmgr_config = [
## pyxpm
# NEH
    { host: base_host, 	   cores: 2, id:"pyxpm-0"  , port:"29451", flags:"s", env:epics_env, cmd:"pyxpm --ip 10.0.1.102 "+pyxpm_optsdb(0)},   # XTPG SXR
    { host: base_host,     cores: 2, id:"pyxpm-1"  , port:"29459", flags:"s", env:epics_env, cmd:"pyxpm --ip 10.0.2.102 "+pyxpm_optsdb(1)},   # RIX XPM
    { host: base_host,     cores: 2, id:"pyxpm-2"  , port:"29453", flags:"s", env:epics_env, cmd:"pyxpm --ip 10.0.3.103 "+pyxpm_optsdb(2)},   # TMO XPM
    { host: base_host,     cores: 2, id:"pyxpm-3"  , port:"29452", flags:"s", env:epics_env, cmd:"pyxpm --ip 10.0.2.103 "+pyxpm_optsdb(3)},   # RIX XPM
    { host: base_host,     cores: 2, id:"pyxpm-4"  , port:"29454", flags:"s", env:epics_env, cmd:"pyxpm --ip 10.0.3.105 "+pyxpm_optsdb(4)},   # TMO XPM
    { host: base_host,     cores: 2, id:"pyxpm-5"  , port:"29456", flags:"s", env:epics_env, cmd:"pyxpm --ip 10.0.1.104 "+pyxpm_optsdb(5)},   # RIX XPM
    { host: base_host,     cores: 2, id:"pyxpm-6"  , port:"29455", flags:"s", env:epics_env, cmd:"pyxpm --ip 10.0.1.105 "+pyxpm_optsdb(6)},   # TMO XPM
    { host: fee_host,      cores: 2, id:"pyxpm-10" , port:"29457", flags:"s", env:epics_env, cmd:"pyxpm --ip 10.0.5.102 "+pyxpm_optsdb(10)},  # FEE XPM
    { host: fee_host,      cores: 2, id:"pyxpm-11" , port:"29450", flags:"s", env:epics_env, cmd:"pyxpm --ip 10.0.5.104 "+pyxpm_optsdb(11)},  # FEE XPM
    { host: gpu_kcu_host,  cores: 2, id:"pyxpm-12" , port:"29451", flags:"s", env:epics_env, cmd:"pykcuxpm --dev /dev/datadev_0 "+pyxpm_optsdb(12)}, # KCU XPM GPU
    { host: base_host,     cores: 2, id:"pyxpm-feh-0"  , port:"29470", flags:"s", env:epics_env, cmd:"pyxpm --ip 10.0.1.107 "+pyxpm_optsdb(0,'FEH')},   # XTPG HXR
    { host: base_host,     cores: 2, id:"pyxpm-feh-1"  , port:"29471", flags:"s", env:epics_env, cmd:"pyxpm --ip 10.0.6.102 "+pyxpm_optsdb(1,'FEH')+" -L"}, # XPM FEH HXR
    { host: base_host,     cores: 2, id:"pyxpm-feh-2"  , port:"29472", flags:"s", env:epics_env, cmd:"pyxpm --ip 10.0.7.102 "+pyxpm_optsdb(2,'FEH')+" -L"}, # XPM MFX HXR
    { host: kcu_host,      cores: 2, id:"pyxpm-feh-3"  , port:"29473", flags:"s", env:epics_env, cmd:"pykcuxpm --dev /dev/datadev_0 "+pyxpm_optsdb(3,'FEH')}, # KCU XPM MFX
    { host: feh_host,      cores: 2, id:"pyxpm-feh-4"  , port:"29474", flags:"s", env:epics_env, cmd:"pyxpm --ip 10.0.8.102 "+pyxpm_optsdb(4,'FEH')}, # XPM FEH XPP
    { host: feh_host,      cores: 2, id:"pyxpm-feh-5"  , port:"29475", flags:"s", env:epics_env, cmd:"pyxpm --ip 10.0.8.103 "+pyxpm_optsdb(5,'FEH')}, # XPM FEH XPP
    { host: xpp_kcu_host,  cores: 2, id:"pyxpm-feh-6"  , port:"29476", flags:"s", env:epics_env, cmd:"pykcuxpm --dev /dev/datadev_0 "+pyxpm_optsdb(6,'FEH')}, # KCU XPM XPP


## EPICS PV exporter
# NEH
 { host: base_host, cores: 1, id:"pvrtmon-0",     port:"29466", flags:"s", env:epics_env, cmd:f"epics_exporter -H tst -I xpm-0 -M {prom_dir} -P DAQ:NEH:XPM:0 -G Us:RxLinkUp,Us:RxDspErrs DeadFrac"},

 { host: base_host, cores: 1, id:"pvrtmon-rix",   port:"29461", flags:"s", env:epics_env, cmd:f"epics_exporter -M {prom_dir} -D rix/xpm-3/DAQ:NEH:XPM:3,rix/xpm-1/DAQ:NEH:XPM:1,rix/xpm-5/DAQ:NEH:XPM:5 -G Us:RxLinkUp,Us:RxDspErrs RunTime Run NumL0Acc L0AccRate NumL0Inp L0InpRate DeadFrac"},

 { host: base_host, cores: 1, id:"pvrtmon-tmo",   port:"29460", flags:"s", env:epics_env, cmd:f"epics_exporter -M {prom_dir} -D tmo/xpm-2/DAQ:NEH:XPM:2,tmo/xpm-4/DAQ:NEH:XPM:4,tmo/xpm-6/DAQ:NEH:XPM:6 -G Us:RxLinkUp,Us:RxDspErrs RunTime Run NumL0Acc L0AccRate NumL0Inp L0InpRate DeadFrac"},

 { host: fee_host,  cores: 1, id:"pvrtmon-fee",   port:"29462", flags:"s", env:epics_env, cmd:f"epics_exporter -M {prom_dir} -D tst/xpm-10/DAQ:NEH:XPM:10,tst/xpm-11/DAQ:NEH:XPM:11,tst/xpm-12/DAQ:NEH:XPM:12 -G Us:RxLinkUp,Us:RxDspErrs RunTime Run NumL0Acc L0AccRate NumL0Inp L0InpRate DeadFrac"},
# FEH
 { host: base_host, cores: 1, id:"pvrtmon-feh-0", port:"29490", flags:"s", env:epics_env, cmd:f"epics_exporter -H txi -I feh-xpm-0 -M {prom_dir} -P DAQ:FEH:XPM:0 -G Cu:RxLinkUp,Cu:RxDspErrs RunTime Run NumL0Acc L0AccRate NumL0Inp L0InpRate DeadFrac"},

 { host: base_host, cores: 1, id:"pvrtmon-mfx",   port:"29491", flags:"s", env:epics_env, cmd:f"epics_exporter -M {prom_dir} -D mfx/feh-xpm-1/DAQ:FEH:XPM:1,mfx/feh-xpm-2/DAQ:FEH:XPM:2,mfx/feh-xpm-3/DAQ:FEH:XPM:3 -G Us:RxLinkUp,Us:RxDspErrs RunTime Run NumL0Acc L0AccRate NumL0Inp L0InpRate DeadFrac"},

 { host: feh_host,  cores: 1, id:"pvrtmon-xpp",   port:"29494", flags:"s", env:epics_env, cmd:f"epics_exporter -M {prom_dir} -D xpp/feh-xpm-4/DAQ:FEH:XPM:4,xpp/feh-xpm-5/DAQ:FEH:XPM:5,xpp/feh-xpm-6/DAQ:FEH:XPM:6 -G Us:RxLinkUp,Us:RxDspErrs RunTime Run NumL0Acc L0AccRate NumL0Inp L0InpRate DeadFrac"},
]
