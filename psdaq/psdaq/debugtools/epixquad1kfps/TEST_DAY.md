# ePix Quad 1 kfps Test-Day Notes

This note records the practical detector-day workflow that was actually useful
for recovering ASIC and bank orientation for `epixquad1kfps`.

## Main Goal

Determine the mapping from logical detector programming coordinates

- ASIC
- bank
- row
- col

to observed `det.raw.raw(evt)` coordinates.

The resulting mapping notes are summarized here:

- https://confluence.slac.stanford.edu/spaces/LCLSIIData/pages/695780057/ePix10ka+Detector+Layout+and+Raw-to-Bank+Mapping

## Important Setup Note

All debugging env vars must be passed through the running `epixquad1kfps` DRP
process environment from the DAQ cnf file, for example `ued.py`.

Setting them only in a client shell is not enough once the DRP process is
already running.

## Debugging Modes

Two modes exist.

### 1. JSON Run Sequence

This mode uses:

- `EPIXQUAD_DEBUG_SEQUENCE_FILE`
- `EPIXQUAD_DEBUG_PATTERN_INDEX`

It is intended for stepping through `sequences/*.json`, often using
`run_pattern_sequence.py`.

Current status:

- supported in code
- may still be broken
- not heavily used on test day

### 2. Direct JSON / NPY

This was the preferred mode and the one that mattered on test day.

#### 2.1 Direct JSON

This mode uses:

- `EPIXQUAD_DEBUG_DIRECT_WRITE_FILE`
- `EPIXQUAD_DEBUG_PATTERN_INDEX`

It writes explicit `(asic, bank, row, col)` operations through the direct-write
path in `epixquad1kfps_config.py`.

#### 2.1.1 Full Bank-0 Fills For 16 ASICs

File used:

- `direct/03_asic_id_bank0_full.json`

Purpose:

- fill logical bank 0 for one ASIC at a time
- identify where each of the 16 ASICs lands in raw detector space

Main detector-day observations:

- this was the main ASIC-location tool
- some bank-0 regions appeared on the right side instead of the left
- that immediately suggested that some banks were effectively rotated by `rot180`

This step gave the coarse ASIC placement.

#### 2.1.2 Sparse Bank 0-3 Markers For ASICs 0-3

Files used:

- `direct/03_bank_markers_178x48.json` for ASIC 0
- `direct/04_bank_markers_178x48_asic1.json` for ASIC 1
- `direct/07_bank_markers_178x48_asic2.json` for ASIC 2
- `direct/08_bank_markers_178x48_asic3.json` for ASIC 3

Purpose:

- probe sparse markers in banks 0, 1, 2, and 3
- determine bank orientation after ASIC location was already known

Important detector-day issue:

- a ghost row was present
- for non-rotated banks, it appeared at row `0`
- for banks rotated by `rot180`, it appeared at row `176`
- this ghost row looked as if its gain-mode bit were also set

That made the first-pass interpretation harder. After ignoring the ghost row,
the sparse marker patterns matched either:

- `identity`
- or `rot180`

cleanly.

This was enough to finish the bank-orientation part of the mapping.

#### 2.1.3 Practical Conclusion

The combination of:

- `direct/03_asic_id_bank0_full.json`
- sparse bank-marker runs for ASICs 0, 1, 2, and 3

gave enough information to construct the correct mapping from logical
`(asic, bank, row, col)` coordinates to `det.raw.raw`.

### 2.2 Direct NPY

This mode uses:

- `EPIXQUAD_DEBUG_MASK_NPY=/path/to/mask.npy`
- `EPIXQUAD_DEBUG_MASK_SELECTED_VALUE=8`
- `EPIXQUAD_DEBUG_MASK_BACKGROUND_VALUE=12`
- `EPIXQUAD_DEBUG_MASK_TRBIT=0`

Use it when you already have a raw detector mask with shape:

- `(4, 352, 384)`

and that mask is already in the same orientation as `det.raw.raw(evt)`.

This mode was not the main driver of the ASIC/bank orientation work, but it is
still useful for direct raw-view experiments.

## Recommended File Order

If the goal is to recover ASIC and bank orientation quickly, the most useful
files are:

1. `direct/03_asic_id_bank0_full.json`
2. `direct/03_bank_markers_178x48.json`
3. `direct/04_bank_markers_178x48_asic1.json`
4. `direct/07_bank_markers_178x48_asic2.json`
5. `direct/08_bank_markers_178x48_asic3.json`

Optional supporting files:

- `direct/01_blind_bank_probe.json`
- `direct/02_blind_bank_probe_rowband.json`
- `direct/04_bank_markers_44x192.json`

## Commands

Example direct JSON setup:

```bash
export EPIXQUAD_DEBUG_DIRECT_WRITE_FILE=psdaq/psdaq/debugtools/epixquad1kfps/direct/03_asic_id_bank0_full.json
export EPIXQUAD_DEBUG_PATTERN_INDEX=0
```

Example direct NPY setup:

```bash
export EPIXQUAD_DEBUG_MASK_NPY=/path/to/mask.npy
export EPIXQUAD_DEBUG_MASK_SELECTED_VALUE=8
export EPIXQUAD_DEBUG_MASK_BACKGROUND_VALUE=12
export EPIXQUAD_DEBUG_MASK_TRBIT=0
```

Remember that these exports must reach the DRP process through the DAQ cnf
environment, not just the launching shell.
