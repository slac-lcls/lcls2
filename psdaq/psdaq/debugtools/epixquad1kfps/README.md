# ePix Quad 1 kfps Debug Test Definitions

This directory holds detector-day test definitions for direct `user.pixel_map`
programming, bypassing the higher-level gainmap conversion workflow.

The intent is to debug the real write/read orientation empirically from sparse
markers in the exact shape consumed by `epixquad1kfps_config.py`:

- `user.pixel_map` shape: `(16, 178, 192)`
- readable ASIC rows: `0..175`
- padded rows: `176..177`

Each test JSON file describes one test group. A loader can:

1. allocate an array of shape `(16, 178, 192)`
2. fill it with `background_value`
3. set each listed marker to `value`
4. program the resulting array as `cfg['user']['pixel_map']`

On top of the per-test JSONs, this directory can also hold pattern-sequence
JSON files. A sequence file provides an ordered list of pattern runs to execute
in one detector session.

## Configure-time debug override

`epixquad1kfps_config.py` can be directed to override `cfg['user']['pixel_map']`
from these JSON definitions at configure time.

Supported environment variables:

- `EPIXQUAD_DEBUG_TEST_FILE=/path/to/test.json`
- `EPIXQUAD_DEBUG_SEQUENCE_FILE=/path/to/sequence.json`
- `EPIXQUAD_DEBUG_DIRECT_WRITE_FILE=/path/to/direct.json`
- `EPIXQUAD_DEBUG_PATTERN_INDEX=<int>` (default `0`)
- `EPIXQUAD_DEBUG_GROUP_INDEX=<int>` (default `0`)
- `EPIXQUAD_DEBUG_MARKER_GROUPS=group1[,group2,...]`
- `EPIXQUAD_DEBUG_PATTERN_OUTDIR=/path/to/save/materialized/patterns`

Use only one of:

- `EPIXQUAD_DEBUG_TEST_FILE`
- `EPIXQUAD_DEBUG_SEQUENCE_FILE`
- `EPIXQUAD_DEBUG_DIRECT_WRITE_FILE`

In standalone test-file mode, if a file contains multiple marker groups, the
loader defaults to group `0` in first-seen order. You can override that with
`EPIXQUAD_DEBUG_GROUP_INDEX`, or select explicit names with
`EPIXQUAD_DEBUG_MARKER_GROUPS`.

When enabled, the configure hook forces:

- `cfg['user']['gain_mode'] = 5`
- `cfg['user']['pixel_map'] = materialized (16,178,192) array`
- per-ASIC `trbit` values from the selected test JSON

In direct-write mode, the injected `user.pixel_map` is only a background-only
placeholder for metadata/debugging. The actual hardware programming is done
later in `config_expert()` from explicit `(asic, bank, row, col)` operations.

Recommended defaults for current tests:

- `background_value = 12`
- `marker_value = 8`
- `trbit_by_asic = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`

These correspond to the direct pixel codes already used in the epixquad
gain-map workflow.

Recommended interpretation for these detector-day tests:

- `background_value = 12` with `trbit = 0` means fixed medium (`FM`)
- `marker_value = 8` means fixed low (`FL`)

That gives a simple fixed-mode test pattern without depending on the auto-mode
raw gain-bit split.

## Gain-mode mapping

For epix10ka / epixquad, the effective mode comes from the combination of:

- `user.pixel_map` code
- ASIC `trbit`
- raw data gain bit for the auto families

| Mode | `user.pixel_map` code | `trbit` | raw gain bit |
|---|---:|---:|---:|
| `FH` | `12` (`0xc`) | `1` | ignored |
| `FM` | `12` (`0xc`) | `0` | ignored |
| `FL` | `8` (`0x8`) | ignored | ignored |
| `AHL_H` | `0` | `1` | `0` |
| `AML_M` | `0` | `0` | `0` |
| `AHL_L` | `0` | `1` | `1` |
| `AML_L` | `0` | `0` | `1` |

For the current JSON test groups, `trbit=0` is recommended because:

- `12` becomes `FM`
- `8` remains `FL`
- both are fixed modes
- the interpretation is simple and does not rely on auto high/medium/low splitting

## Test JSON schema

Each test JSON uses the following top-level fields:

- `version`: schema version
- `test_name`: stable short name
- `description`: brief purpose of the test
- `array_shape`: always `[16, 178, 192]`
- `readable_rows`: always `176`
- `background_value`: fill value for all pixels not explicitly marked
- `default_marker_value`: default value for markers if omitted
- `trbit_by_asic`: list of 16 per-ASIC `trbit` values
- `markers`: list of marker points

Each marker entry contains:

- `label`: human-readable identifier
- `group`: group name used to associate points that belong to the same marker
- `asic`: ASIC index in `user.pixel_map`
- `row`: row in `user.pixel_map`
- `col`: column in `user.pixel_map`
- `value`: pixel code to write for this point
- `tags`: optional tags for quadrant / hypothesis / purpose

## Pattern-sequence JSON schema

Each sequence JSON uses the following top-level fields:

- `version`: schema version
- `sequence_name`: stable short name
- `description`: brief purpose of the sequence
- `defaults`: suggested scan defaults
- `patterns`: ordered list of pattern runs

The `defaults` block can include:

- `readout_count`: suggested number of events per pattern
- `record`: whether runs should be recorded
- `config_alias`: suggested DAQ config alias

Each pattern entry contains:

- `pattern_index`: integer pattern number
- `test_file`: path to one test JSON file relative to this directory
- `marker_groups`: one or more marker groups from that test file to enable for this run
- `group_index`: optional numeric fallback if `marker_groups` is omitted; defaults to `0`
- `label`: short human-readable pattern label
- `purpose`: what this pattern is expected to prove
- `priority`: optional relative priority such as `high`, `medium`, `low`
- `readout_count`: optional per-pattern override
- `notes`: optional list of free-form notes

## Files

- `manifest.json`: quick index of all test files
- `tests/01_module0_asic_orientation.json`
- `tests/02_all_modules_anchor_asic0.json`
- `tests/03_single_asic_quadrants.json`
- `tests/04_single_asic_column_regions.json`
- `tests/05_single_asic_row_regions.json`
- `sequences/priority_shortlist.json`
- `sequences/detector_day_v1.json`

The first goal is to establish the empirical mapping from programmed
`(asic,row,col)` coordinates to observed raw gain-bit locations, without
assuming the current bank math is correct.

## Direct register-write JSON schema

Direct register-write mode is intended for blind bank tests after coarse module
and ASIC orientation are known. It bypasses the current logical `pixel_map`
reconstruction and writes ASIC-local coordinates directly.

Top-level fields:

- `version`: schema version
- `direct_name`: stable short name
- `description`: brief purpose of the direct-write file
- `coordinate_mode`: optional logical coordinate convention; defaults to `bank_rc_178x48`
- `background_value`: scalar or 16-element list, one background code per ASIC
- `default_selected_value`: default explicit-write value
- `trbit_by_asic`: scalar or 16-element list of ASIC `trbit` values
- `patterns`: ordered list of direct-write patterns

Each direct-write pattern contains:

- `pattern_index`: integer pattern number
- `label`: short human-readable pattern label
- `ops`: list of direct hardware write operations

Supported direct op kinds:

- `pixel`
  - fields: `asic`, `bank`, `row`, `col`, optional `value`
  - writes one explicit ASIC-local pixel
- `bank_fill`
  - fields: `asic`, `bank`, optional `row_range`, optional `col_range`, optional `value`
  - fills a rectangular region inside one explicit bank
  - defaults depend on `coordinate_mode`

Supported `coordinate_mode` values:

- `bank_rc_178x48`
  - logical bank shape: `178 x 48`
  - logical `bank` means hardware bank id
  - logical `row`: `0..177`
  - logical `col`: `0..47`
  - hardware write:
    - `RowCounter <- row`
    - `ColCounter <- BANK_OFFSETS[bank] | col`

- `bank_rc_44x192`
  - logical bank shape: `44 x 192`
  - logical `bank` means one 44-row band inside the ASIC
  - logical `row`: `0..43`
  - logical `col`: `0..191`
  - hardware expansion:
    - `hw_row = bank * 44 + row`
    - `hw_bank = col // 48`
    - `hw_col = col % 48`
    - `RowCounter <- hw_row`
    - `ColCounter <- BANK_OFFSETS[hw_bank] | hw_col`

In both modes:

- `asic`: `0..15`
- `bank`: `0..3`

and the direct path bypasses the existing `mrow/mcol` reconstruction logic.

Example file:

- `direct/01_blind_bank_probe.json`
- `direct/02_blind_bank_probe_rowband.json`
