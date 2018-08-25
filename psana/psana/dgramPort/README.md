# Converting XTC to XTC2 with Python

Note: there are currently two approaches for how we do this.

* for more complex data (e.g. configure transitions) with small numbers of events we go through json files on disk using Eliseo's scripts
* for simpler data with many events we use a ZMQ version that Chuck modified.

In principle we could merge these, but time does not currently allow.

## Eliseo Notes: How XTC to XTC2 Translation Scripts Work

Run "parse_lcls1_data.py" using LCLS-I psana. It will parse LCLS-I XTC files from four different
example experiments (including the crystallography data) and write out json files. Using LCLS-II psana, run "translate_xtc_json.py".  This will read in the json files and write them to XTC2 files. 

Serialization using json is usually much faster than pickle, but it can't handle numpy
arrays. The first script converts the arrays to lists of [bytes, shape, type], which are
then reconstituted. 

## A Second Email Message From Eliseo:

An example parsing an LCLS-I XTC file:

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

## Additions From Chuck

Eliseo's approach doesn't work with bulk data because the json files becomes too large.  Chuck has modified versions (parse_lcls1_cxic0415.py, parse_lcls1_cxic0515.py, translate_xtc_demo.py) that don't save json to disk, instead transmitting the json data over ZMQ to translate_xtc_demo.py.
