# DAQ control

## What it is

A DAQ "platform" is a set of processes (DRPs, trigger and monitoring event
builders, ...) driven by one **control** process, the `CollectionManager` in
`psdaq/psdaq/control/control.py`. The control process runs a state machine,
sends each transition to the other processes over ZMQ, collects their replies,
and inserts the corresponding timing transitions through the XPM. Clients such
as `daqstate`, the control GUI, and Python scripts built on `DaqControl` talk
to the control process to query and change its state.

Everything on this page needs a running DAQ (hardware or a simulated
platform); none of it is usable with offline data.

## Key concepts

### States and transitions

From `ControlDef` (`psdaq/psdaq/control/ControlDef.py`):

- states: `reset`, `unallocated`, `allocated`, `connected`, `configured`,
  `starting`, `paused`, `running`
- transitions: `rollcall`, `alloc`, `dealloc`, `connect`, `disconnect`,
  `configure`, `unconfigure`, `beginrun`, `endrun`, `beginstep`, `endstep`,
  `enable`, `disable`, `slowupdate`, `reset`

The state machine (built with the `transitions` package in
`CollectionManager.__init__`) connects them as follows; `reset` goes from any
state to `reset`, and `slowupdate` is allowed in `running` without changing
state:

```text
reset/unallocated --rollcall--> unallocated --alloc--> allocated --connect--> connected
connected --configure--> configured --beginrun--> starting --beginstep--> paused --enable--> running
running --disable--> paused --endstep--> starting --endrun--> configured --unconfigure--> connected
connected --disconnect--> allocated --dealloc--> unallocated
```

A client can ask for a target state (`setstate.<state>`); the control process
then walks the needed transitions one by one.

What some transitions do (from `control.py`):

- `rollcall`: broadcast to all processes listening on the platform; collects
  who is there (level, alias, host, pid).
- `alloc`: selects the processes for this partition. It fails unless at least
  one DRP and one TEB are present ("at least one DRP is required", "at least
  one TEB is required") and only warns if there is no MEB ("ami NOT supported
  in absence of MEB").
- `configure` and later: sent to the partition; replies from the DRP, TEB and
  MEB processes are collected. Configure also fetches the trigger
  configuration (`-t TRIGGER_CONFIG`) from the configuration database.

### Ports and messages

`ControlDef.PORT_BASE = 29980`. Port functions in `ControlDef.py`, all
`PORT_BASE + platform + offset`:

| Function | Offset | Socket in the control process |
|---|---|---|
| `back_pull_port` | 0 | PULL: replies from DAQ processes |
| `back_pub_port` | 10 | PUB: transitions to DAQ processes (topics `all`, `partition`) |
| `front_rep_port` | 20 | REP: client requests |
| `front_pub_port` | 30 | PUB: status, errors, progress to clients |
| `fast_rep_port` | 40 | REP: quick requests (record flag, instrument, ...) |
| `step_pub_port` | 50 | PUB: step-done notifications |

Messages are JSON dicts `{'header': {'key', 'msg_id', 'sender_id'}, 'body': {...}}`
built by `create_msg()` in `ControlDef.py`. On the C++ side the DAQ processes
derive from `CollectionApp` (`psdaq/psdaq/service/Collection.hh`, base port
`zmq_base_port = 29980`), which subscribes to `all`, adds `partition` after
`alloc`, and routes `configure`, `beginrun`, `enable`, ... to
`handlePhase1` (see [DRP and event building](drp-eventbuilding.md)).

### Client API: `DaqControl`

`psdaq/psdaq/control/DaqControl.py`:
`DaqControl(*, host, platform, timeout)` connects to the control process on
`host` for `platform` (timeout in ms). Methods:

| Method | Request sent |
|---|---|
| `getState()` | `getstate`; returns the state name, or `'error'` |
| `getStatus()` | `getstatus`; returns a tuple (transition, state, config alias, recording, ...) |
| `getPlatform()`, `selectPlatform(body)` | `getstate` / `selectplatform` |
| `getInstrument()` | `getinstrument` (fast socket) |
| `setState(state, phase1Info={})` | `setstate.<state>` |
| `setTransition(transition, phase1Info={})` | the transition name |
| `setConfig(config)` | `setconfig.<alias>` |
| `setRecord(recordIn)` | `setrecord.1` / `setrecord.0` (fast socket) |
| `monitorStatus()` | blocks until the next status/error/progress/step message |

The `set*` methods return an error string, or `None` on success.

### Command-line tools

Entry points from `psdaq/pyproject.toml` (`[project.scripts]`):

| Command | Module | What it does |
|---|---|---|
| `control` | `psdaq.control.control:main` | The control process itself (`-P` instrument, `-u` unique id, `-C` default config alias are required; `-x` master XPM and `-B` PV base unless `--sim`). |
| `daqstate` | `psdaq.control.daqstate:main` | Get or set the state: `--state`, `--transition`, `--monitor`, `--config`, `--record`, `--bypass`, `-B` (with `-p` platform, `-P` instrument, `-C` collection host). |
| `selectPlatform`, `showPlatform` | `psdaq.control.*` | Select processes for the partition; show the platform. |
| `timed_run` | `psdaq.control.timed_run:main` | Record a run of `--duration` seconds, using a `.cnf` file. |
| `getrun`, `currentexp` | `psdaq.control.*` | Current run / experiment from the run-control web service. |
| `control_gui` | `psdaq.control_gui.app.control_gui:control_gui` | Qt control GUI. |

`TimedRun`, `ConfigScan` and `BlueskyScan` (same directory) are library
classes used by such scripts; `BlueskyScan` is a DAQ device for the bluesky
RunEngine. `TimedRun` has an API page:
[psdaq.control.TimedRun](../api/python/timedrun.md).

### Launching a platform

The processes are listed in a `.cnf` file (examples in `psdaq/psdaq/cnf/`,
e.g. `tmo.cnf`, `rix.cnf`, `lab3-base.cnf`) and started with `procmgr`
(`psdaq/psdaq/procmgr/procmgr`; usage
`procmgr { start | stop | stopall | restart | status } configfile ...`). Each
entry gives the host, an id, flags and a command; the `u` and `p` flags
append `-u <id>` and `-p <platform>`. An excerpt of `tmo.cnf`:

```python
{ host: collect_host, id:'control', flags:'spu', env:epics_env,
  cmd:f'control -P {hutch} -B DAQ:NEH -x 2 -C BEAM {auth} {url} -d {cdb}/configDB -t trigger -S 1 -T 40000 -V {elog_cfg}'},
{ host: 'drp-srcf-cmp030', id:'teb0', flags:'spu', cmd:teb_cmd},
```

`psdaq/psdaq/slurm/` provides an alternative launcher based on Slurm
(`daqmgr` command).

## Minimal example

Record a timed run from Python, adapted from
`psdaq/psdaq/control/lab3_timed_run.py` (needs a running platform). `TimedRun`
reads `args.v` in its constructor, so the parser must define `-v`; see the
[psdaq.control.TimedRun](../api/python/timedrun.md) API page:

```python
import argparse
import sys

from psdaq.control.DaqControl import DaqControl
from psdaq.control.TimedRun import TimedRun

parser = argparse.ArgumentParser()
parser.add_argument('-p', type=int, choices=range(0, 8), default=1, help='platform')
parser.add_argument('-C', metavar='COLLECT_HOST', required=True, help='collection host')
parser.add_argument('-t', type=int, metavar='TIMEOUT', default=10000, help='timeout msec')
parser.add_argument('-v', action='store_true', help='be verbose')
parser.add_argument('--duration', type=int, default=10, help='run duration seconds')
args = parser.parse_args()

control = DaqControl(host=args.C, platform=args.p, timeout=args.t)
daqState = control.getState()
if daqState == 'error':
    sys.exit('failed to get initial DAQ state')

run = TimedRun(control, daqState=daqState, args=args)
run.stage()
run.set_running_state()
run.sleep(args.duration)
run.unstage()
run.push_socket.send_string('shutdown')   # stop TimedRun's helper thread
```

From the shell, the same control process can be driven with `daqstate`, for
example `daqstate -p <platform> -P <instrument> -C <collection host> --state running`.

## Where in the code

- `psdaq/psdaq/control/control.py`: `CollectionManager`, `DaqXPM`, `DaqPVA`, `RunParams`, `main`.
- `psdaq/psdaq/control/ControlDef.py`: states, transitions, ports, message helpers.
- `psdaq/psdaq/control/DaqControl.py`: client API.
- `psdaq/psdaq/control/daqstate.py`, `timed_run.py`, `TimedRun.py`, `ConfigScan.py`, `BlueskyScan.py`: clients and helpers.
- `psdaq/psdaq/control_gui/`: GUI.
- `psdaq/psdaq/procmgr/`, `psdaq/psdaq/cnf/`, `psdaq/psdaq/slurm/`: launching.
- `psdaq/psdaq/service/Collection.hh`: the C++ side of the protocol (`CollectionApp`).

API pages: [psdaq.control.control](../api/python/control.md),
[psdaq.control.DaqControl](../api/python/daqcontrol.md),
[psdaq.control.TimedRun](../api/python/timedrun.md).
