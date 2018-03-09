# ami
The LCLS-II online graphical analysis monitoring package.

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
Or for mpi:
```mpirun -n 3 ami-worker --mpi static://examples/worker.json```

Then start a manager with a specified configuration:
```ami-manager```

Finally, start a GUI (client):
```python ami/ami/client.py```

You should see an interactive QT window. There is also a `dummy_gui.py` that gives just text output.

# Status/To-do

2/9/18

backend
-------
* integration with EPICS
* should we reuse MPI?
    - yes. bigger question: should we maintain a ZMQ message passing mode
    - mpi event-builder (timeout)
    - send/receive for the graph (handle the case where the different clients are in inconsistent graph states)
* throttling [pick-N pattern, done on a per-heartbeat basis] (e.g. for visualizing images)
    - use top-down approach with round-robin based on heartbeat counting.  collector has to: avg/sum, scatter plot, plot vs. time/event number, plot most recent image.  
    - Use gather/reduce pattern, but implement with send/receive plus timeout (either with mpi_probe, or heartbeat counting and discard old). 
* Fix bug with first stage gather where collector will mess up if one worker is too far behind
    - make sure we don't get 'off by 1'
    - need a way to agree on the phases for the heartbeats, timing system should have some sort of number for these
* Fault tolerance (tradeoff: avoid complicating code too much which can create more failures, and cause the code to be difficult to understand)
* External interfaces:
    - hutch python - epics / json documents broker of blue sky
    - DRP feedback?
* only reconfigure graph on transitions
* manager needs to discover when the configuration has failed to apply on one of the workers
* need to fix gui and manager now that we are using MPI too
    
    
frontend
--------
* how to select multiple inputs into an analysis box
* how to choose between two graph options (e.g. roi vs. projection)
* Ability to edit the graph graphically.
* Visualization of the graph
* duplicate non-broken AMI functionality


other
-----
* cache worker config in manager pub so workers get it again on restart
* autosave
* read real XTC
* Minimal viable documentation ( so Dan can remember what is going on )
