# amii
This is the LCLS-II online analysis monitoring package, a.k.a. the LCLS-II
version of AMI. Currently this is nothing more than a prototype / platform for
spitballing ideas.

# Requirements
* Python 2.7 or 3.5+
* ipython
* pyzmq
* numpy

All these requirements are subject to change at any time!

# Examples
If you use the setup.py included to set this up you should now have two console
scripts available on your path: `ami-worker` and `ami-client`. Several example
configuration files are included in the examples directory.

To run ami with three workers run the following:
```ami-worker -n 3```

Currently their is only a simple IPython client for talking to ami:
```ami-client examples/rate.json```

Once the IPython session loads run `send_cfg()` in the terminal to send the
configuration to workers and start ami.

# Status/To-do

5/18/17
Factored OpConfig out of Operation in anticipation of editing it

Created an ami-manager which can be queried to get the config

used code below to request config from mgr, but now want to create an ROIConfig.

How do we do this, since the required parameters are in the ROIOperation class?

Do we need an ROIConfig class?  We were hoping to avoid this to make operation creation easy for users.

To do:

- multiple outputs of boxes going to different boxes (e.g. for laser-on/laser-off filtering)
- make the graph-editing in dummy_gui.py "real" (i.e. actually run events thru system and change graph)

```
Running the system:

ami-worker -n 1
ami-client examples/roi.json
send_cfg()

to see a picture
python ami/client/gui.py

to run the manager:
ami-manager examples/roi.json

to ask the manager for the graph and modify it:
python ami/client/dummy_gui.py
```
