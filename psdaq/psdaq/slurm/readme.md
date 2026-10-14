Slurm for DAQ

Use slurm to  start, stop, and gather output from DAQ processes.  

Process log location:
- `daqmgr` writes process logs under `<output_root>/YYYY/MM`.
- `--output` (or `-o`) sets `<output_root>`.
- If `--output` is not provided, default `<output_root>` is:
  - `~/daq/logs` for `xppopr`
  - `~` for other hutches/users

List of CLIs:
- taskset -c 4-63 drp -P tst -C drp-srcf-cmp035 -M /cds/group/psdm/psdatmgr/etc/config/prom/tst -d /dev/datadev_1 -o /cds/data/drpsrcf -k batching=yes,directIO=yes -l 0x1 -D ts
- drp -P tmo -D ts ..
- control -P tmo ...
- on localhost: control_gui -P ...

First stage:
- Keep node allocations hardwired
