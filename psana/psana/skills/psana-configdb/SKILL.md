---
name: psana-configdb
description: Read-only lookups against the LCLS-II ConfigDB web service for detector/system configuration. Use for "was detector X misconfigured", "what is the trigger/timing configuration", "list configdb devices/aliases for a hutch", "did a config parameter change mid-run", "configdb history".
---

# Skill: psana-configdb

# LCLS-II ConfigDB Read-Only Lookup

You are querying the ConfigDB web service to read detector/trigger/timing
configuration for a hutch. This skill is strictly READ-ONLY — you observe
configuration, you never modify it.

**Related skills:** if you arrived here without first checking metrics or
logs, consider loading `psana-daq-monitor` or `psana-daq-logs` first (or
`psana-daq` if the report is still vague) to narrow down which detector or
device's configuration is actually relevant. If you need to correlate a
config change with a specific run's wall-clock time, load `elog-search`
(`skill(name="elog-search")`) to look up that run's start/end time in the
LCLS eLog — do not guess at eLog query syntax from this skill.

---

## No authentication required for reads

The psdaq `configdb.py` web-service client (`psdaq/psdaq/configdb/configdb.py`,
class `configdb`) normally talks to an authenticated endpoint using
`ws-auth` in the URL plus HTTP Basic Auth via the `CONFIGDB_AUTH` environment
variable (see `_get_response`, `psdaq/psdaq/configdb/configdb.py:55-83`).
However, the bundled `configdb ls` CLI subcommand does this at
`psdaq/psdaq/configdb/configdb.py:577`:

    # authentication is not required, adjust url accordingly
    url = args.url.replace('ws-auth', 'ws').replace('ws-kerb', 'ws')

Simply substituting `ws-auth` → `ws` (or `ws-kerb` → `ws`) in the URL drops
all authentication requirements for READ operations. This has been verified
live — with no credentials, no Kerberos ticket, and `CONFIGDB_AUTH` unset —
against the real production service. Use this base URL:

    https://pswww.slac.stanford.edu/ws/configdb/ws/configDB/

Note: `ws`, NOT `ws-auth`.

---

## Verified working endpoints

All of these are plain unauthenticated `GET` requests. Recommend `curl` or
Python `requests` directly against the base URL above — there is no need to
invoke the psdaq CLI script or import `psdaq`.

| Endpoint | Description | Verified sample response |
|---|---|---|
| `GET .../get_hutches/` | List all hutches | `{"status_code": 200, "success": true, "msg": "OK", "value": ["ued", "tmo", "rix", "tst", "asc", "txi", "mfx", "xpp", "det", "TMO"]}` |
| `GET .../get_aliases/<hutch>/` | List aliases for a hutch (e.g. `xpp`) | `{"status_code": 200, "success": true, "msg": "OK", "value": ["BEAM"]}` |
| `GET .../get_devices/<hutch>/<alias>/` | List devices for a hutch/alias (e.g. `xpp/BEAM`) | `{"status_code": 200, "success": true, "msg": "OK", "value": ["epix100_0", "epixuhr3x2_0", "epixuhr3x2_1", "hsd_0", "hsd_1", "hsd_2", "hsd_3", "jungfrau1M_0", "jungfrau1M_1", "jungfrau_0", "jungfrau_1", "timing_0", "timing_1", "trigger_0", "wav8_ipm2_0", "wav8_ipm3_0", "wav8_lodcm_0", "wav8_user_0"]}` |
| `GET .../get_configuration/<hutch>/<alias>/<device>/` | Full device configuration (e.g. `xpp/BEAM/timing_0`) | Large nested config dict with fields like `alg:RO`, `firmwareBuild:RO`, `help:RO`, etc. — not a small fixed payload, inspect the returned JSON directly. |
| `GET .../get_history/<hutch>/<alias>/<device>/` (JSON array body required) | Track whether specific config parameter(s) changed over time (e.g. `xpp/BEAM/timing_0`) | See below — this is not a plain parameterless GET. |

Unlike the other endpoints above, `get_history` requires a JSON array body
naming which dot-separated parameter(s) to track, with the first component
being the device config name. Verified live against the production service:

    curl -G https://pswww.slac.stanford.edu/ws/configdb/ws/configDB/get_history/xpp/BEAM/timing_0/ \
         -d '["detName:RO"]'

Verified sample response (abbreviated):

    {"status_code": 200, "success": true, "msg": "OK",
     "value": [{"date": "2025-10-01T18:20:01.927000+00:00", "key": 1, "detName:RO": "timing_0"},
               {"date": "2025-10-17T21:58:18.277000+00:00", "key": 2, "detName:RO": "timing_0"}, ...]}

Each entry has `date` (UTC ISO8601) and `key` (an integer config version
number).

**Gotcha:** calling `get_history` via plain `GET` with no body returns
`{"success": false, "msg": "get_history: no POST data"}` — the JSON body is
mandatory even though the HTTP verb is `GET` (the `curl -G ... -d '...'`
idiom sends the `-d` payload as a URL-encoded query string on a GET
request, despite the error message's wording).

### Equivalent read-only CLI

If `psdaq` happens to be on `PYTHONPATH`, the equivalent read-only CLI
commands are:

    configdb ls <hutch>[/<alias>]
    configdb cat <hutch>/<alias>/<device>

These are not required — the raw HTTP GETs above are the primary interface
for this skill.

---

## Correlating history with a specific run

`get_history` gives `{date, key}` pairs in UTC — it does **not** give run
numbers directly.

To answer "was detector X misconfigured during run N":

1. First get that run's start/end wall-clock time from the LCLS eLog. Load
   the sibling `elog-search` skill for that lookup — invoke via
   `skill(name="elog-search")`. Do not guess at eLog query syntax from this
   skill; `elog-search` is a separate skill package with its own documented
   query interface.
2. Once you have the run's UTC time window, find which `get_history` entry's
   `date` falls within `[run start, run end)` — that entry's `key` is the
   config version active during the run.
3. To inspect that historical config's actual content: **this is
   UNVERIFIED.** `get_configuration`'s documented signature takes an alias
   name, not a numeric key, and this workflow has not been live-tested for
   fetching a specific historical key's full config content. Verify this
   live before relying on it — do not assume `get_configuration` (or any
   other endpoint) can fetch an arbitrary historical `key`'s content just
   because `get_history` returned that key.

## Diagnosing a symptom via config

Two-stage model to keep in mind when a detector's data looks wrong:

- ConfigDB tells you what configuration was **REQUESTED**.
- psana's own `_configs`/`_seg_configs()` attributes — populated from XTC
  Config transitions baked into the actual data file/shared-memory stream —
  tell you what configuration **ACTUALLY REACHED** the data for a given
  run. These can diverge from ConfigDB's current value if a config change
  happened between runs with no new Configure transition.

`psana/psana/app/config_dump.py` and `psana/psana/detector/cfg_utils.py`
(`dump_det_config`/`dump_seg`/`dumpvars`) are the tools for inspecting what
actually reached a run's data.

See `reference/device-config-diagnostics.md` (in this skill directory) for
which specific ConfigDB fields matter for which symptoms, per device type —
including which fields psana never reads at all.

---

## Hard guardrail

**NEVER use a URL containing `ws-auth` or `ws-kerb`, and NEVER set or read
the `CONFIGDB_AUTH` environment variable.** This skill is READ-ONLY by
design. Do not attempt `add_alias`, `add_device_config`, `modify_device`,
`remove_device`, `rename_device`, `transfer_config`, or any other mutating
operation — all of these require the authenticated `ws-auth` endpoint and
would change live experiment configuration.

If a question requires a write (e.g. "fix the misconfigured device"), stop
and tell the user this is outside the scope of this read-only skill.
