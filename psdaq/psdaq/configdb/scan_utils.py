import json
import pprint
pp = pprint.PrettyPrinter(indent=4)

#
#  scan_utils.py
#
def copy_config_entry(d,old,key):
    o = old
    keys = key.split('.')
    for k in keys[:-1]:
        if k in o:
            o = o[k]
        else:
            print('Error in lookup of [:types:]{:}'.format(key))
            raise KeyError
        if k in d:
            d = d[k]
        else:
            d[k] = {}
            d = d[k]
    try:
        d[keys[-1]] = o[keys[-1]]
    except:
        pp.pprint(o.keys())
        pp.pprint(d.keys())
        print('Caught exception on {}'.format(key))
        raise KeyError(keys[-1])

def update_config_entry(r,old,update):
    # Still need special handling for enums
    if not ':types:' in r:
        r[':types:'] = {}
    for key in update.keys():
        if key == 'step_docstring':
            r[':types:'][key] = 'CHARSTR'
        elif key == 'step_value':
            r[':types:'][key] = 'FLOAT'
        else:
            copy_config_entry(r[':types:'],old[':types:'],key)

    for key,value in update.items():
        d = r
        keys = key.split('.')
        for k in keys[:-1]:
            if k in d:
                d = d[k]
            else:
                d[k] = {}
                d = d[k]
        d[keys[-1]] = value
        sval = '{:}'.format(value)
        if len(sval)<64:
            print('[{:}] : {:}'.format(key,sval))
        else:
            print('[{:}] : (truncated)'.format(key))


def copy_reconfig_keys(r,old,update):
    # Still need special handling for enums
    if not ':types:' in r:
        r[':types:'] = {}
    for key in update:
        if key == 'step_docstring':
            r[':types:'][key] = 'CHARSTR'
        elif key == 'step_value':
            r[':types:'][key] = 'FLOAT'
        else:
            copy_config_entry(r[':types:'],old[':types:'],key)

    for key in update:
        if key == 'step_docstring':
            r[key] = ''
        elif key == 'step_value':
            r[key] = 0.
        else:
            copy_config_entry(r,old,key)

def check_json_keys(chk, ref, exact=False):
    """Checks that keys found in chk are also in ref and return True if so.

    chk and ref are lists of strings of JSON.  Lists lengths must be the same.
    """
    if not isinstance(chk, list):
        print(f'chk is not a list; got {type(chk)}')
        return False
    if not isinstance(ref, list):
        print(f'ref is not a list; got {type(ref)}')
        return False
    if len(chk) != len(ref):
        print(f'Input list lengths don\'t match: {len(chk)} vs {len(ref)}')
        return False

    def _count(d):
        cnt = 0
        for key in d.keys():
            cnt += 1
            if isinstance(d[key], dict):
                cnt += _count(d[key])
        return cnt

    def _check(cd, rd, idx):
        cnt = 0
        for key in cd.keys():
            if key not in rd.keys():
                print(f'Key {key} not found in reference dict[{idx}]')
                return 0
            cnt += 1
            if isinstance(cd[key], dict):
                if not isinstance(rd[key], dict):
                    print(f'Key {key} found in reference dict[{idx}] but is not a dict')
                    return 0
                c = _check(cd[key], rd[key], idx)
                if c == 0:
                    print(f'Key check failed at {key}')
                    return 0
                cnt += c
        return cnt

    for idx, elm in enumerate(chk):
        cd = json.loads(elm)
        rd = json.loads(ref[idx])
        cnt = _check(cd, rd, idx)
        if cnt == 0:
            print(f'Key check failed in level {idx}')
            print(f'check     dict[{idx}]:', cd)
            print(f'reference dict[{idx}]:', rd)
            return False
        if exact:
            c = _count(rd)
            if cnt != c:
                print(f'All keys were found in the reference dict[{idx}], '
                      'but the number of keys in each was different: {cnt} vs {c}')
                print(f'check     dict[{idx}]:', cd)
                print(f'reference dict[{idx}]:', rd)
                return False
    return True
