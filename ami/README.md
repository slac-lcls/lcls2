# ami
This is the LCLS-II online analysis monitoring package, a.k.a. the LCLS-II
version of AMI. Currently this is nothing more than a prototype / platform for
spitballing ideas.

# Requirements
* Python 3.5+
* ipython
* pyzmq
* numpy

All these requirements are subject to change at any time!

# Examples
If you use the setup.py included to set this up you should now have two console
scripts available on your path: `ami-worker` and `ami-manager`. Several example
configuration files are included in the examples directory.

To run ami with three workers run the following in lcls2/ami:
```ami-worker -n 3 static://examples/worker.json```

Then start a manager with a specified configuration:
```ami-manager```

Finally, start a GUI (client):
```python ami/ami/client.py```

You should see an interactive QT window. There is also a `dummy_gui.py` that gives just text output.

# Status/To-do

2/9/18

* Ability to edit the graph graphically.
* Visualization of the graph
* Minimal viable documentation ( so Dan can remember what is going on )
* Fault tolerance
* External interfaces:
    - hutch python - epics / json documents broker of blue sky
    - DRP feedback?
* scaling of Qt in general
* duplicate non-broken AMI functionality
* autosave
* read real XTC
* only reconfigure graph on transitions
* cache worker config in manager pub so workers get it again on restart
* manager needs to discover when the configuration has failed to apply on one of the workers
* Fix bug with first stage gather where collector will mess up if one worker is too far behind
    - make sure we don't get 'off by 1'
    - need a way to agree on the phases for the heartbeats - timing system should have some sort of number for these
