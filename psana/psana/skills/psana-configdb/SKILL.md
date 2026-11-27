---
name: psana-configdb
description: Read-only lookups against the LCLS-II ConfigDB web service for detector/system configuration. Use for "was detector X misconfigured", "what is the trigger/timing configuration", "list configdb devices/aliases for a hutch", "did a config parameter change mid-run", "configdb history".
---

# Skill: psana-configdb — LCLS-II ConfigDB Read-Only Lookup

Queries the ConfigDB web service to read detector/trigger/timing
configuration for a hutch. Strictly read-only — observes configuration,
never modifies it.

**Related skills:** if arrived here without first checking metrics or
logs, consider `psana-daq-monitor`/`psana-daq-logs` first (or `psana-daq`
if the report is still vague) to narrow which device's configuration is
relevant. To correlate a config change with a specific run's wall-clock
time, load `elog-search` by name — don't guess at its query syntax here.

**Arrived here from another skill?** The hand-off must include a
**configuration alias**, not just a hutch and detector — every endpoint
below is scoped by alias (`<hutch>/<alias>/<device>`), and a hutch
commonly has more than one. **Never pick the first alias returned by
`get_aliases` as a default.** Use the handed-off alias; if none was given,
ask which alias is active rather than guessing — the wrong one silently
inspects configuration that was never active for the session in question.
See `psana-daq/SKILL.md`'s "Configuration alias" section for the full
fallback chain (`control.log`'s `Loaded configDb` line, then `teb0.log`'s
`Trigger loaded from` line, then `control.log`'s `CMDLINE:` `-C` flag as
a launch-time-only fallback).

## No authentication required for reads

The bundled `configdb ls` CLI substitutes `ws-auth`→`ws` (or `ws-kerb`→
`ws`) in the URL to drop authentication for reads
(`psdaq/psdaq/configdb/configdb.py:577`). Verified live — no credentials,
no Kerberos ticket, `CONFIGDB_AUTH` unset — against the real production
service:

    https://pswww.slac.stanford.edu/ws/configdb/ws/configDB/

Note `ws`, not `ws-auth`.

## Verified working endpoints

Plain unauthenticated `GET` requests — use `curl`/`requests` directly, no
need for the psdaq CLI or importing `psdaq`.

| Endpoint | Description |
|---|---|
| `GET .../get_hutches/` | List all hutches |
| `GET .../get_aliases/<hutch>/` | List aliases for a hutch |
| `GET .../get_devices/<hutch>/<alias>/` | List devices for a hutch/alias |
| `GET .../get_configuration/<hutch>/<alias>/<device>/` | Full device configuration — large nested dict, inspect the returned JSON directly |
| `GET .../get_history/<hutch>/<alias>/<device>/` (JSON array body) | Parameter change history — see below, not a plain parameterless GET |

`get_history` requires a JSON array body naming which dot-separated
parameter(s) to track:

```bash
curl --request GET --fail \
  https://pswww.slac.stanford.edu/ws/configdb/ws/configDB/get_history/xpp/BEAM/timing_0/ \
  --header 'Content-Type: application/json' \
  --data '["detName:RO"]'
```

`--fail` matters: without it, `curl` exits `0` even on an HTTP error
response (a `4xx`/`5xx` body still "succeeded" as a transfer) — a script
checking only the exit code would read a real failure as a successful,
empty-ish response.

Response: `{"value": [{"date": "2025-10-01T18:20:01.927+00:00", "key": 1,
"detName:RO": "timing_0"}, ...]}` — `date` (UTC ISO8601), `key` (integer
config version).

**The discriminator is the `Content-Type: application/json` header, not
the HTTP verb or `-G`.** The service reports `"get_history: no POST data"`
on a `GET` with the array attached via `-G`/query-string or via a bare
`--data` with no `Content-Type` header — in both cases it never inspects
the query string for this endpoint, and without the header it doesn't
recognize the body as data at all, regardless of HTTP method. **Do not use
`curl -G -d`** — that idiom sends the array as a URL query string, which
this endpoint ignores entirely. Use `--request GET --data '[...]' --header
'Content-Type: application/json'` as shown above.

### Equivalent read-only CLI

If `psdaq` is on `PYTHONPATH`: `configdb ls <hutch>[/<alias>]`, `configdb
cat <hutch>/<alias>/<device>`. Not required — the raw HTTP GETs above are
the primary interface.

## History entries are candidate database edits, not application events

`get_history` records when a value in the **database** changed — not when
or whether that change took effect in a run. An entry's `date` can precede
`BeginRun` by any amount of time, and a later edit may never have reached
the DAQ at all if no Configure transition followed it. Do not treat "the
history entry nearest a run's start" as automatically "the version that
was active" without the check below.

To answer "was detector X misconfigured during run N":

1. Get the run's start/end wall-clock time from the LCLS eLog (load
   `elog-search`).
2. Sort `get_history`'s entries by `date`. **The active version is the
   latest entry with `date <= run.start`** — not the entry nearest the
   run in either direction, and specifically not an entry with `run.start
   < date < run.end` (that's a change *during* the run, not the version
   active when it began).
3. **Corroborate against recorded content when available** — psana's own
   `_configs`/`_seg_configs()` (populated from XTC Config transitions)
   records what was actually in the data stream for that run, which is
   stronger evidence than a database timestamp alone that a given version
   was really the one in effect.
4. If any entry satisfies `run.start < date < run.end`, the run spans a
   **mid-run config change** — report this explicitly: divide the run
   into intervals bounded by each such entry's `date`, and state which
   `key` applied to each interval, rather than naming one version for the
   whole run.
5. **Historical-key retrieval is unsupported by the endpoint this skill
   uses.** `get_configuration` takes an alias string, not a historical
   key — there is no documented way to fetch the full content of a past
   `key` through the unauthenticated `ws` read endpoint. If a specific
   historical version's full content is needed, say this limitation
   explicitly rather than attempting a key-based `get_configuration` call
   or implying one would work.

## Diagnosing a symptom via config

Two-stage model: ConfigDB tells you what was **requested**. psana's own
`_configs`/`_seg_configs()` — from XTC Config transitions in the actual
data stream — tell you what was **recorded** for a run, which can diverge
from ConfigDB's current value (a config change between runs with no new
Configure transition) and is itself one step removed from hardware truth
(recorded, not independently confirmed to have taken effect).
`psana/psana/app/config_dump.py` and `psana/psana/detector/cfg_utils.py`
(`dump_det_config`/`dump_seg`/`dumpvars`) inspect what was recorded for a
run's data.

See `reference/device-config-diagnostics.md` for which specific ConfigDB
fields matter for which symptoms, per device type — including which
fields have no named-algorithm psana read at all (not the same as "no
psana tool can show the value" — a generic dump tool still can).

## Hard guardrail

**Never use a URL containing `ws-auth` or `ws-kerb`, and never set or read
`CONFIGDB_AUTH`.** Never attempt `add_alias`, `add_device_config`,
`modify_device`, `remove_device`, `rename_device`, `transfer_config`, or
any other mutating operation — all require the authenticated `ws-auth`
endpoint and would change live experiment configuration. If a question
requires a write ("fix the misconfigured device"), stop and say this is
outside this read-only skill's scope.
