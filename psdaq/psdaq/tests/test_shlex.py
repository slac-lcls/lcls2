import shlex

#
# fix_env_whitespace
#
# Replace ',' with ' ' in environment string
#
def fix_env_whitespace(ss):
    print(f"*** enter fix_env_whitespace({ss})")
    if len(ss) == 0:
        return ss
    result = list()
    qq = shlex.split(ss)
    print(f"*** qq={qq}")
    for ii in qq:
        print(f"  *** ii={ii}")
        if len(ii) == 0:
            continue
        rr = ii.split('=')
        print(f"  ### len(rr)={len(rr)} rr={rr}")
        if len(rr) != 2:
            print(f"  *** ERR: required format: X1=Y1 [X2=Y2 [...]]")
            continue
        if ',' in rr[1]:
            ee = '='.join([rr[0], shlex.quote(rr[1].replace(',', ' '))])
        else:
            ee = ii
        result.append(ee)
    return " ".join(result)

def fix_env_whitespace_verbose(ss):
    print(f' input: {ss}')
    output = fix_env_whitespace(ss)
    print(f'output: {output}')
    return output

CONDA_PREFIX = '/cds/sw/ds/ana/conda2/inst/envs/ps-4.6.1'

def run():

    print("Test 1")
    fix_env_whitespace_verbose('X1=Y1 X2=Y2')

    print("Test 2")
    fix_env_whitespace_verbose('A=B C=D,E F=G')

    print("Test 3")
    fix_env_whitespace_verbose(f'EPICS_PVA_ADDR_LIST=172.21.140.55,127.0.0.1 EPICS_PVA_AUTO_ADDR_LIST=YES LD_LIBRARY_PATH={CONDA_PREFIX}/epics/lib/linux-x86_64:{CONDA_PREFIX}/pcas/lib/linux-x86_64')

    print("Test 4")
    fix_env_whitespace_verbose('EPICS_PVA_ADDR_LIST=172.21.151.255,127.0.0.1 LD_LIBRARY_PATH=/cds/sw/ds/ana/conda2/inst/envs/ps-4.6.1/epics/lib/linux-x86_64:/cds/sw/ds/ana/conda2/inst/envs/ps-4.6.1/pcas/lib/linux-x86_64')

if __name__ == '__main__':
    run()
