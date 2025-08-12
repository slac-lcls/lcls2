import argparse
import os
import sys

def main():

    parser = argparse.ArgumentParser(description='create .py for timing system monitoring')

    parser.add_argument('-o', help='output file (daqmgr input file)', metavar='CNF_FILE', default='tdet-base.py')
    parser.add_argument('-x', help='xpm daqmgr input file', nargs='+', type=str)
    parser.add_argument('-e', help='expt daqmgr input file', nargs='+', type=str)

    args = parser.parse_args()
    print(args)

    xpms = []
    for a in args.x:
        print(f'--{a}--')
        sys.path.append(os.path.dirname(a))
        config_dict = {"platform": None, "station": 0, "config": None}
        exec(compile(open(a).read(), a, "exec"),{},config_dict)

        xparser = argparse.ArgumentParser()
        xparser.add_argument('--ip',default='0.0.0.0')
        xparser.add_argument('-P',default=None)

        for entry in config_dict['procmgr_config']:
            prog = entry['cmd'].split(' ',1)[0]
            if prog=='pyxpm' or prog=='pykcuxpm':
                x = xparser.parse_known_args(entry['cmd'].split(' '))[0]
                xpms.append(f'{x.ip},{x.P}')

    drps = []
    fims = []
    hsds = []

    for a in args.e:
        print(f'--{a}--')
        sys.path.append(os.path.dirname(a))
        config_dict = {"platform": None, "station": 0, "config": None}
        exec(compile(open(a).read(), a, "exec"),{},config_dict)

        xparser = argparse.ArgumentParser()
        xparser.add_argument('-D',default=None)
        xparser.add_argument('-k',default=None)

        for entry in config_dict['procmgr_config']:
            prog = entry['cmd'].strip().split(' ',1)[0]
            if "host" in entry:
                host = entry["host"]
                if prog=='taskset':
                    prog = entry['cmd'].strip().split(' ',4)[3]
                if prog=='drp':
                    x = xparser.parse_known_args(entry['cmd'].split(' '))[0]
                    if x.D in ('ts',):
                        drps.append((host,'tdet',host.upper()))
                    elif x.D in ('piranha4','opal',):
                        drps.append((host,'clnk',host.upper()))
                    elif x.D in ('epix100',):
                        drps.append((host,'e100',host.upper()))
                    elif x.D in ('jungfrau',):
                        drps.append((host,'ludp',host.upper()))
                    elif x.D in ('wave8',):
                        for k in x.k.split(','):
                            p = k.split('=')
                            if p[0]=='epics_prefix':
                                fims.append(p[1])
                                break
                    elif x.D in ('hsd',):
                        for k in x.k.split(','):
                            p = k.split('=')
                            if p[0]=='hsd_epics_prefix':
                                hsds.append(p[1])
                                break
                elif prog in ('epicsArch','drp_pva','drp_bld',):
                    drps.append((host,'tdet',host.upper()))

    with open(args.o,'w') as f:
        header = ''
        header += 'import os\n\n'
        header += 'if not platform: platform = "6"\n\n'
        header += 'CONDA_PREFIX = os.environ.get("CONDA_PREFIX","")\n'
        header += 'host, cores, id, flags, env, cmd, rtprio, port = ("host", "cores", "id", "flags", "env", "cmd", "rtprio", "port")\n'
        header += 'ld_lib_path = f"LD_LIBRARY_PATH={CONDA_PREFIX}/epics/lib/linux-x86_64:{CONDA_PREFIX}/pcas/lib/linux-x86_64"\n'
        header += 'epics_env = f"EPICS_PVA_ADDR_LIST=172.21.159.255 {ld_lib_path}"\n\n'
        header += "tdet_opts = '--type DrpTDet  --dev /dev/datadev_1'\n"
        header += "clnk_opts = '--type Lcls2Pgp --dev /dev/datadev_0 --pgp4 False'\n"
        header += "e100_opts = '--type Lcls2Epix --dev /dev/datadev_0 --pgp4 True'\n"
        header += "ludp_opts = '--type Lcls2Udp --dev /dev/datadev_0'\n\n"
        header += "procmgr_config = [\n"
        f.write(header)

        for a in set(drps):
            f.write(" { "+f"host: '{a[0]}', cores: 2, id:'pytdrp-{a[0][-3:]}', port:'0', flags:'s', env:epics_env, cmd:'pytdrp '+{a[1]}_opts"+" },\n")

        xpmarg = ' '.join(set(xpms))
        fimarg = ' '.join(set(fims))
        hsdarg = ' '.join(set(hsds))

        f.write(" { "+f"host: drp-srcf-mon001, cores: 2, id:'pytdet-001', port:'0', flags:'s', env:epics_env, cmd:'pytdet_collector --xpm {xpmarg} --fim {fimarg} --hsd {hsdarg}"+" },\n")

        f.write("]\n")
        f.close()

if __name__=='__main__':
    main()

