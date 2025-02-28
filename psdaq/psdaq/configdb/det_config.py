import pyrogue as pr
import numpy as np
from collections import deque,OrderedDict
import logging

def mode(a):
    uniqueValues = np.unique(a).tolist()
    uniqueCounts = [len(np.nonzero(a == uv)[0])
                    for uv in uniqueValues]

    modeIdx = uniqueCounts.index(max(uniqueCounts))
    return uniqueValues[modeIdx]

def dumpvars(prefix,c):
    print(prefix)
    for key,val in c.nodes.items():
        name = prefix+'.'+key
        dumpvars(name,val)

def retry(cmd,val=None):
    itry=0
    while(True):
        try:
            if val is None:
                cmd()
            else:
                cmd(val)
        except Exception as e:
            logging.warning(f'Try {itry} of {cmd}({val}) failed.')
            if itry < 3:
                itry+=1
                continue
            else:
                raise e
        break

def _dict_compare(d1,d2,path):
    for k in d1.keys():
        if k in d2.keys():
            if d1[k] is dict:
                _dict_compare(d1[k],d2[k],path+'.'+k)
            elif (d1[k] != d2[k]):
                print(f'key[{k}] d1[{d1[k]}] != d2[{d2[k]}]')
        else:
            print(f'key[{k}] not in d1')
    for k in d2.keys():
        if k not in d1.keys():
            print(f'key[{k}] not in d2')


#
#  Helper function for calling underlying pyrogue interface
#
def intToBool(d,types,key):
    if isinstance(d[key],dict):
        for k,value in d[key].items():
            intToBool(d[key],types[key],k)
    elif types[key]=='boolEnum':
        d[key] = False if d[key]==0 else True

def ordered(d,order):
    od = OrderedDict()
    for key in order:
        od[key] = d[key]
    return od

#
#  Translate a dictionary of register value pairs to a yaml file for rogue configuration
#
def dictToYaml(d,types,keys,dev,path,name,tree,ordering=None):
    v = {'enable':True}
    for key in keys:
        if key in d:
            if ordering is None:
                v[key] = d[key]
            else:
                try:
                    v[key] = ordered(d[key],ordering[key])
                except KeyError:
                    print('*** key:', key)
                    print('*** d:', d)
                    print('*** o:', ordering)
                    raise
            intToBool(v,types,key)
            v[key]['enable'] = True
        else:
            v[key] = {'enable':False}

    key = tree[-1]
    nd = OrderedDict()
    nd[key] = v
    for leaf in reversed(tree[:-1]):
        nd = {leaf:{'enable':True,'ForceWrite':False,'InitAfterConfig':False,key:nd[key]}}
        key = leaf

#    nd = {'ePixHr10kT':{'enable':True,'ForceWrite':False,'InitAfterConfig':False,'EpixHR':v}}
    yaml = pr.dataToYaml(nd)
    fn = path+name+'.yml'
    f = open(fn,'w')
    f.write(yaml)
    f.close()
    setattr(dev,'filename'+name,fn)
    print(f'Wrote {fn}')

    #  Need to remove the enable field else Json2Xtc fails
    for key in keys:
        if key in d:
            try:
                del d[key]['enable']
            except KeyError:
                pass
    return fn

#
#  Apply the configuration dictionary to the rogue registers
#
def apply_dict(pathbase,base,cfg):
    rogue_translate = {}
    rogue_translate['TriggerEventBuffer0'] = 'TriggerEventBuffer[0]'
    rogue_translate['TriggerEventBuffer1'] = 'TriggerEventBuffer[1]'

    depth = 0
    my_queue  =  deque([[pathbase,depth,base,cfg]]) #contains path, dfs depth, rogue hiearchy, and daq configdb dict tree node
    while(my_queue):
        path,depth,rogue_node, configdb_node = my_queue.pop()
        if(dict is type(configdb_node)):
            for i in configdb_node:
                k = rogue_translate[i] if i in rogue_translate else i
                try:
                    my_queue.appendleft([path+"."+i,depth+1,rogue_node.nodes[k],configdb_node[i]])
                except KeyError:
                    logging.info('Lookup failed for node [{:}] in path [{:}]'.format(i,path))

        #  Apply
        if('get' in dir(rogue_node) and 'set' in dir(rogue_node) and path != pathbase ):
            if False:
                logging.info(f'NOT setting {path} to {configdb_node}')
            else:
                print(f'Setting {path} to {configdb_node}')
                #logging.info(f'Setting {path} to {configdb_node}')
                retry(rogue_node.set,configdb_node)

