# Eliseo notes for how this software works

Run "parse_lcls1_data.py" from the old psana. It will parse LCLS-I xtc files from four different
example experiments (including the crystallography data) and write out json files. "translate_xtc_demo.py"
will read in the json files and write them to XTC2 files. 

Serialization using json is usually much faster than pickle, but it can't handle numpy
arrays. The first script converts the arrays to lists of [bytes, shape, type], which are
then reconstituted. 

And from a second email message from Eliseo:

here's parsing an LCLS1 xtc file


```
ds = DataSource('exp=cxid9114:run=89')
source = Source('DetInfo(CxiDs1.0:Cspad.0)')
detector = CsPad.DataV2
config = CsPad.ConfigV5

cfgd = parse_dgram(ds, source, detector, config, event_limit)
with open("crystal_dark.json", 'w') as f:
f.write(json.dumps(cfgd.events))
```

To save in the xtc format, I reference the experiment by the name of the json file. 

```translate_xtc_demo('crystal_dark', 2)```

This writes a pseudo-configure file in crystal_dark_configure.xtc and the events data in crystal_dark_evts.xtc. 

The examples required some amount of hard-coding values, so I'd be surprised if the translation
works for an arbitrary LCLS1 xtc file.
