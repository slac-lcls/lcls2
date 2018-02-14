import sys
sys.path.append('/reg/neh/home/yoon82/temp/lcls2/build/psana')
sys.path.append('/reg/neh/home/yoon82/temp/lcls2/psana')
from DataSource import DataSource
from Detector import Detector

ds = DataSource('/reg/neh/home/yoon82/temp/lcls2/hsd_121517.xtc', verbose=1, debug=0)
det0 = Detector('hsd1', ds.config)

for evt in ds:
    channels = det0.raw(evt)
    break

print("HSD channels:")
print(channels)

from IPython import embed
embed()
exit()

#print("hsd1: ", ds.config.hsd1.hsd1.software, ds.config.hsd1.hsd1.version)
#print("raw: ", ds.config.hsd1.raw.software, ds.config.hsd1.raw.version)

reader = None
plug = None
mysoftware = []

def children(grandparent, parent):
    tree.append(parent)
    _parent = getattr(grandparent, parent)
    try:
        if "software" in vars(_parent) and "version" in vars(_parent):
            print("Found: ", tree, getattr(_parent, "software"), getattr(_parent, "version"))
            mysoftware.append(getattr(_parent, "software"))
            tree.pop()
        else:
            for i, child in enumerate(vars(_parent)):
                children(_parent, child)
    except:
        pass

tree = []
for detname in vars(ds.config):
    children(ds.config, detname)

print("mysoftware: ", mysoftware)
if "hsd" in mysoftware:
    from hsd import Hsd # TODO: this does not use the version information
    reader = Hsd("hsd","1.2.3")

for ii, evt in enumerate(ds):
    # hsd_raw = evt.hsd_1.chan0
    chan0 = reader.readRaw(evt.hsd1.raw.chan0, verbose=1)
    print(chan0)
    if ii == 0: break


#print("ds._config refcount:",sys.getrefcount(ds._config))
#print("evt refcount:",sys.getrefcount(evt))











def children(parent):
    for child in vars(parent):
        _obj = getattr(parent, child)
        try:
            children(_obj)
        except:
            for key in vars(_obj):
                val = getattr(_obj, key)
                print(child, key, val)

for detname in vars(ds.config):
    print("@@@ detname: ", detname)
    _det = getattr(ds.config, detname)
    children(_det)
exit()

for detname in vars(ds.config):
    print("detname: ", detname)
    _det = getattr(ds.config, detname)
    for dataName in vars(_det): # hsd1, raw
        print("### dataName: ", detname, dataName)
        # Look for dataNames requiring hsd software
        _data = getattr(_det, dataName)
        for data in vars(_data):  # hsd1, raw
            print("vars: ", detname, dataName, data)

            #from hsd import Hsd  # this needs to be the ds.config.software:hsd and version:1.2.3


            #reader = Hsd(ds.config.hsd1.raw.software, ds.config.hsd1.raw.version)  # dataName: raw

#if "detname" in ds.config.hsd1.raw.software:
#    from hsd import Hsd # this needs to be the ds.config.software:hsd and version:1.2.3
#    reader = Hsd(ds.config.hsd1.raw.software, ds.config.hsd1.raw.version) # dataName: raw
    #reader.add_dataName(ds.config.dataName)





