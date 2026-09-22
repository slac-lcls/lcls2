---
name: psana-configdb
description: Read-only lookups against the LCLS-II ConfigDB web service for detector/system configuration. Use for "was detector X misconfigured", "what is the trigger/timing configuration", "list configdb devices/aliases for a hutch", "did a config parameter change mid-run", "configdb history".
---

# Skill: psana-configdb

# LCLS-II ConfigDB Read-Only Lookup

You are querying the ConfigDB web service to read detector/trigger/timing
configuration for a hutch. This skill is strictly READ-ONLY — you observe
configuration, you never modify it.

**Related skills:** reuse the hutch, alias/device, session/run, UTC window and
available evidence passed by the caller. Load logs or metrics only if needed to
localize the question. `elog-search` is an optional external skill, not bundled
here: use it only if installed and relevant run times are not already supplied.
Otherwise use retained run/transition evidence or report run bounds unavailable.

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

`get_history` takes a JSON array of parameter names in a **GET request body**,
as implemented by `configdb.get_history()` and `_get_response()` in
`psdaq/psdaq/configdb/configdb.py` (`requests.get(..., json=plist)`). Example:

```bash
curl --fail --silent --show-error --max-time 20 --request GET \
  --header 'Content-Type: application/json' \
  --data '["detName:RO"]' \
  'https://pswww.slac.stanford.edu/ws/configdb/ws/configDB/get_history/<hutch>/<alias>/<device>/'
```

Substitute established identifiers before use. Do not use `curl -G -d`: it
moves data into the query string and does not match the client contract. This
example is source-checked and tested with a synthetic local HTTP receiver,
not newly validated against production. A proxy may reject GET bodies; report
that access failure instead of silently changing the request semantics.

Inspect both HTTP status and the JSON `success` field before using `value`.
History entries may include UTC `date`, integer `key` and requested values;
a successful empty result is different from an error or unavailable history.

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

1. Establish the run and relevant Configure/scan-step context from supplied run
   metadata, retained control logs or XTC transitions. Resolve time zones and
   uncertainty; a launch prefix/mtime interval is not an exact run window.
2. Treat history records as **database-change candidates**, not application
   events. Include the latest applicable entry at or before Configure and
   earlier entries if history is incomplete, plus changes during the requested
   interval. A configuration may have been established long before BeginRun.
   A later edit may never have reached the DAQ. Even the latest preceding key
   is only a candidate without corroboration.
3. Corroborate the applied identity with run-associated Configure content,
   recorded key (when present), device/segment/alias and release evidence.
   `get_config.py::get_config_with_params` retrieves by alias and may resolve
   `_cfgTypeRef`; the alias alone is not an immutable version. Separate what
   was requested, what was recorded and what hardware behavior demonstrates.
   Partial Configure fields need not uniquely identify a database key.
4. **Historical-key retrieval is unsupported by this skill's verified
   interface.** The in-tree client concatenates an alias string into the
   `get_configuration` path; it does not establish arbitrary numeric-key
   retrieval. Do not invent an endpoint or substitute today's alias contents.
   If key/content retrieval is unavailable, say so and use retained Configure
   evidence with explicit limits. No production probe is required to complete
   an evidence-limited historical report.

## Diagnosing a symptom via config

Two-stage model to keep in mind when a detector's data looks wrong:

- ConfigDB tells you what configuration was **REQUESTED**.
- psana's own `_configs`/`_seg_configs()` attributes — populated from XTC
  Config transitions baked into the actual data file/shared-memory stream —
  tell you what configuration was **RECORDED** in the data for a given
  run. Recorded fields are stronger run-associated evidence than today's alias,
  but do not independently prove hardware applied every setting. These can
  diverge from ConfigDB's current value if a config change
  happened between runs with no new Configure transition.

`psana/psana/app/config_dump.py` and `psana/psana/detector/cfg_utils.py`
(`dump_det_config`/`dump_seg`/`dumpvars`) are the tools for inspecting what
actually reached a run's data.

See [reference/device-config-diagnostics.md](reference/device-config-diagnostics.md) for
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
