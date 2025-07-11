import numpy

def dumpvars(prefix,c):
    for key,val in vars(c).items():
        name = prefix+'.'+key
        if (isinstance(val,int) or
            isinstance(val,float) or
            isinstance(val,str) or
            isinstance(val,list) or
            isinstance(val,numpy.ndarray)):
            print('{:} {:}'.format(name,val))
        elif hasattr(val,'value'):
            print('{:} {:}'.format(name,val.names[val.value]))
        else:
            try:
                dumpvars(name,val)
            except:
                print('Error dumping {:} {:}'.format(name,type(val)))

def dump_seg(seg,cfg):
    print('-- segment {:} --'.format(seg))
    dumpvars('config',cfg.config)

def dump_det_config(det,name):
    for config in det._configs:
        if not name in config.__dict__:
            print('Skipping config {:}'.format(config.__dict__))
            continue
        scfg = getattr(config,name)
        for seg,segcfg in scfg.items():
            dump_seg(seg,segcfg)
