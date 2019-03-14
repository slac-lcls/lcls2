from psana.dgramPort.typed_json import cdict
from psana import DataSource
import numpy as np
import subprocess

top = cdict()
top.setInfo("hsd", "xpphsd", "serial1234", "No comment")
top.setAlg("hsdConfig", [0,0,1])
# Scalars
top.define_enum("EnableEnum", {"Disable": 0, "Enable": 1})
#top.set("top.enable", 1, "EnableEnum")
top.set("enable", 1, "EnableEnum")

top.set("raw.start", 4, "UINT16")
top.set("raw.gate", 20, "UINT16")
top.set("raw.prescale", 1, "UINT16")

top.set("fex.start", 4, "UINT16")
top.set("fex.gate", 20, "UINT16")
top.set("fex.prescale", 1, "UINT16")
top.set("fex.ymin", 0, "UINT16")
top.set("fex.ymax", 2000, "UINT16")
top.set("fex.xpre", 1, "UINT16")
top.set("fex.xpost", 1, "UINT16")

top.writeFile("hsd_config.json")
fname = "hsd_config.xtc2"
subprocess.call(["json2xtc", "hsd_config.json", fname])

ds = DataSource(fname)
myrun = next(ds.runs())
hsdcfg = myrun.configs[0].xpphsd[0]
