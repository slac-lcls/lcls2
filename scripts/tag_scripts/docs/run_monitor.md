# `run_monitor.sh`

The script cron runs. It runs one job type (`branch` or `tag`) for `lcls2`, for every monitored hutch, one after another, and handles logging, locking and failure email.

```
./run_monitor.sh [--dry-run] <branch|tag> [hutch ...]
```

| Argument | Meaning |
|---|---|
| `branch` | Run `branch_out.sh` |
| `tag` | Run `single_push_collective_tag.sh` |
| `hutch ...` | Optional. Hutches to run for; default is the `HUTCHES` list in the script. |
| `--dry-run` | Passed to the job scripts. Nothing is committed, tagged or pushed, and no email is sent. |

Exit status is `0` if every job succeeded, `1` otherwise.

---

## Why a wrapper

- **One cron line per job, however many hutches.** Without it, each hutch needs its own branch and tag crontab lines. With six hutches that's 12 lines, and adding a hutch means editing the crontab.
- **The branch and tag repos are shared by every hutch.** If two jobs used the same repo at the same time, they would switch branches under each other and hit git's lock files. The wrapper runs hutches one at a time and holds a lock so two runs can never overlap.
- **Logs and email in one place.** Each hutch gets a dated log per run, and a run sends at most one email listing everything that failed.

---

## Walkthrough

### 1. Configuration

```bash
PROJECT=lcls2
PREFIX=lcls

REL="${REL:-/sdf/group/lcls/ds/ana/sw/conda2/rel}"

HUTCHES=(xpp)

LOG_BASE="${LOG_BASE:-$REL/cron_logs}"
LOG_KEEP_DAYS=90
LOCK_WAIT="${LOCK_WAIT:-7200}"
MAIL_TO="${MAIL_TO-mavaylon@slac.stanford.edu}"
```

| Setting | Meaning |
|---|---|
| `PROJECT` | The project this copy runs: `lcls2`, the repo it lives in. Its shared repos are `$REL/branch_repo_lcls2/lcls2` and `$REL/tag_repo_lcls2/lcls2`. |
| `PREFIX` | Production clones handled: directories in `$REL/<hutch>` whose names start with `lcls`. |
| `REL` | Root of the release area. Production clones are in `$REL/<hutch>`. |
| `HUTCHES` | Hutches run by default. Add a hutch here to bring it under monitoring. |
| `LOG_BASE` | Where logs go. |
| `LOG_KEEP_DAYS` | Run logs older than this are deleted. |
| `LOCK_WAIT` | Seconds to wait for an earlier run to finish (2 hours). |
| `MAIL_TO` | Failure email address. Set it to empty (`MAIL_TO=`) to turn off email. |

`PROJECT` and `PREFIX` are the only lines that differ between the copies of this script kept in other repos. Everything else, including `branch_out.sh` and `single_push_collective_tag.sh`, is identical.

`REL`, `LOG_BASE`, `LOCK_WAIT` and `MAIL_TO` can also be set from the environment. That's handy for testing against a copy. `MAIL_TO` uses `${MAIL_TO-…}` (no colon), so setting it to an *empty* value really turns email off instead of falling back to the default.

```bash
export PATH="/usr/bin:/bin:$PATH"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
```

cron starts jobs with a very short `PATH`, so `git`, `flock` and `mail` are made findable explicitly. `SCRIPT_DIR` is the folder the wrapper lives in. It calls the two job scripts from the same folder, so all three always come from the same checkout and version.

### 2. Arguments

The first non-flag argument is the job (`branch` or `tag`); anything after it is a list of hutches. With no hutches given, `HUTCHES` is used. `--dry-run` can go anywhere.

### 3. The lock

```bash
exec 9> "$LOG_BASE/.run_monitor.lock"
if ! flock -w "$LOCK_WAIT" 9; then
    ... send_mail "Monitor ${PROJECT} ${JOB} job skipped" ...
    exit 1
fi
```

- `exec 9> file` opens the lock file as file descriptor 9 for the rest of the script.
- `flock -w 7200 9` takes an exclusive lock on it, waiting up to 2 hours if another run holds it.
- The lock is released automatically when the script exits, even if it crashes.

It's a single lock file in `LOG_BASE`, so it covers both job types, and it is also shared with the monitor jobs of any other project that uses the same `LOG_BASE`. If another run is still going when this one starts, this one waits. If the lock is still held after `LOCK_WAIT`, something is stuck: the run is skipped and an email is sent.

### 4. The loop

```bash
for hutch in "${RUN_HUTCHES[@]}"; do
    job_dir="$LOG_BASE/$hutch/${PROJECT}_${JOB}"
    log="$job_dir/${RUN_STAMP}.log"
    root_dir="$REL/$hutch"
```

For each hutch it runs the job against the project's shared repo (`REPO`, the branch or tag repo, chosen once before the loop):

```bash
if [ "$JOB" = "branch" ]; then
    LOG_ROOT="$LOG_BASE/$hutch" "$SCRIPT_DIR/branch_out.sh" "${DRY_RUN_ARGS[@]}" \
        "$hutch" "$root_dir" "$REPO" "$PREFIX" > "$log" 2>&1
else
    "$SCRIPT_DIR/single_push_collective_tag.sh" "${DRY_RUN_ARGS[@]}" \
        "$hutch" "$root_dir" "$REPO" "$PREFIX" > "$log" 2>&1
fi
```

- Each hutch's full output goes to its own log file.
- For the branch job, `LOG_ROOT` is set to the hutch's log folder. `branch_out.sh`'s failure reports then land next to that hutch's run logs, in `<hutch>/lcls2_branch/failed_runs/`.
- If `$REL/<hutch>` doesn't exist (for example, a typo in `HUTCHES`), that hutch fails without creating any log folders.
- After each hutch, run logs older than `LOG_KEEP_DAYS` in that folder are deleted. `-maxdepth 1` keeps `failed_runs/` out of it, so failure reports are kept until someone removes them.

### 5. Log layout

```
cron_logs/
  .run_monitor.lock
  run_monitor_lcls2.log                  ← wrapper summaries (cron appends here)
  xpp/
    lcls2_branch/2026-09-26_020000.log
    lcls2_branch/failed_runs/…_FAILED.log
    lcls2_tag/…
  tmo/
    …
```

Each run gets a new, timestamped file. Earlier versions wrote to one fixed file per job and overwrote it every run, which is how the cause of a failure on 2026-09-13 was lost.

### 6. Summary and email

```
FAILED  xpp  /…/cron_logs/xpp/lcls2_branch/2026-09-26_020000.log
ok      tmo  /…/cron_logs/tmo/lcls2_branch/2026-09-26_020000.log
Details:
  xpp:
      failed: lcls2_060226: failed: git push -q -u origin xpp-lcls2_060226 (git@github.com: Permission denied (publickey).; fatal: Could not read from remote repository.)
      skipped: lcls2_031125: no clone entry in reflog
=== finished …: 1 failed ===
```

One line per hutch goes to stdout, and cron appends it to `run_monitor_lcls2.log`.

**Details** name the hutch, the clone and the reason for every failed or skipped clone, so you can see what went wrong without opening the logs. After each hutch, `summary_items` reads the job log's `=== Summary ===` section, which both job scripts end with, and turns each listed clone into a `failed: <clone>: <reason>` or `skipped: <clone>: <reason>` line:

```bash
summary_items() {
    awk '
        /^=== Summary ===/        { in_summary = 1; next }
        !in_summary               { next }
        /^Failed:/                { section = "failed"; next }
        /^Skipped:/               { section = "skipped"; next }
        /^[A-Za-z]/               { section = ""; next }
        section && /- /           { sub(/^(### )?[[:space:]]*- /, ""); print section ": " $0 }
    ' "$1"
}
```

If a hutch failed but its log lists no failed clone, the problem was with the shared repo itself (for example `Base repo is not clean`) or the script stopped early. In that case the last 4 non-empty lines of the log are shown instead, prefixed `log:`.

If anything failed, one email goes to `MAIL_TO` with the failed hutches and their logs, the same **Details**, and all results, and the script exits `1`. Skipped clones appear in the details but never cause an email on their own: an old clone whose reflog has expired would otherwise trigger one every week. No email is sent in a dry run.

---

## Common tasks

**Check what a run would do** (safe; changes nothing):
```
./run_monitor.sh --dry-run branch
./run_monitor.sh --dry-run tag
```

**Try a hutch before adding it:**
```
./run_monitor.sh --dry-run branch tmo
./run_monitor.sh --dry-run tag tmo
```

**Run one hutch by hand** (for real):
```
./run_monitor.sh branch tmo
```
