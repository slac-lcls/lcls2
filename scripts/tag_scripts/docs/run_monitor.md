# `run_monitor.sh`

The script cron runs. It runs one job type (`branch` or `tag`) for every monitored hutch, for both lcls2 and ami, one after another, and handles logging, locking and failure email.

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

- **One cron line per job, however many hutches.** Without it, each hutch needs four crontab lines (lcls2/ami × branch/tag). With six hutches that's 24 lines, and adding a hutch means editing the crontab.
- **The four branch/tag repos are shared by every hutch.** If two jobs used the same repo at the same time, they would switch branches under each other and hit git's lock files. The wrapper runs everything one at a time and holds a lock so two runs can never overlap.
- **Logs and email in one place.** Each job gets a dated log, and a run sends at most one email listing everything that failed.

---

## Walkthrough

### 1. Configuration

```bash
REL="${REL:-/sdf/group/lcls/ds/ana/sw/conda2/rel}"

HUTCHES=(xpp)

LOG_BASE="${LOG_BASE:-$REL/cron_logs}"
LOG_KEEP_DAYS=90
LOCK_WAIT="${LOCK_WAIT:-7200}"
MAIL_TO="${MAIL_TO-mavaylon@slac.stanford.edu}"
```

| Setting | Meaning |
|---|---|
| `REL` | Root of the release area. Production clones are in `$REL/<hutch>`, and the shared repos are in `$REL/branch_repo_*` and `$REL/tag_repo_*`. |
| `HUTCHES` | Hutches run by default. Add a hutch here to bring it under monitoring. |
| `LOG_BASE` | Where logs go. |
| `LOG_KEEP_DAYS` | Run logs older than this are deleted. |
| `LOCK_WAIT` | Seconds to wait for an earlier run to finish (2 hours). |
| `MAIL_TO` | Failure email address. Set it to empty (`MAIL_TO=`) to turn off email. |

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
    ... send_mail "Monitor $JOB job skipped" ...
    exit 1
fi
```

- `exec 9> file` opens the lock file as file descriptor 9 for the rest of the script.
- `flock -w 7200 9` takes an exclusive lock on it, waiting up to 2 hours if another run holds it.
- The lock is released automatically when the script exits, even if it crashes.

It's a single lock for both job types. So if a branch run is still going when the tag run starts, the tag run waits for it. If the lock is still held after `LOCK_WAIT`, something is stuck: the run is skipped and an email is sent.

### 4. The loop

```bash
for hutch in "${RUN_HUTCHES[@]}"; do
    for prefix in lcls ami; do
        ...
        job_dir="$LOG_BASE/$hutch/${project}_${JOB}"
        log="$job_dir/${RUN_STAMP}.log"
        root_dir="$REL/$hutch"
```

For each hutch it runs the job for lcls2 and then ami, choosing the matching shared repo:

```bash
if [ "$JOB" = "branch" ]; then
    LOG_ROOT="$LOG_BASE/$hutch" "$SCRIPT_DIR/branch_out.sh" "${DRY_RUN_ARGS[@]}" \
        "$hutch" "$root_dir" "$REL/branch_repo_${project}/${project}" "$prefix" > "$log" 2>&1
else
    "$SCRIPT_DIR/single_push_collective_tag.sh" "${DRY_RUN_ARGS[@]}" \
        "$hutch" "$root_dir" "$REL/tag_repo_${project}/${project}" "$prefix" > "$log" 2>&1
fi
```

- Each job's full output goes to its own log file.
- For the branch job, `LOG_ROOT` is set to the hutch's log folder. `branch_out.sh`'s failure reports then land next to that hutch's run logs, in `<hutch>/<project>_branch/failed_runs/`.
- If `$REL/<hutch>` doesn't exist (for example, a typo in `HUTCHES`), that entry fails without creating any log folders.
- After each job, run logs older than `LOG_KEEP_DAYS` in that folder are deleted. `-maxdepth 1` keeps `failed_runs/` out of it, so failure reports are kept until someone removes them.

### 5. Log layout

```
cron_logs/
  .run_monitor.lock
  run_monitor.log                      ← wrapper summaries (cron appends here)
  xpp/
    lcls2_branch/2026-09-26_020000.log
    lcls2_branch/failed_runs/…_FAILED.log
    ami_branch/…
    lcls2_tag/…
    ami_tag/…
  tmo/
    …
```

Each run gets a new, timestamped file. Earlier versions wrote to one fixed file per job and overwrote it every run, which is why the cause of the 2026-09-13 failure was lost.

### 6. Summary and email

```
ok      xpp lcls2  /…/cron_logs/xpp/lcls2_branch/2026-09-26_020000.log
ok      xpp ami    /…/cron_logs/xpp/ami_branch/2026-09-26_020000.log
=== finished …: 0 failed ===
```

One line per hutch/project goes to stdout, and cron appends it to `run_monitor.log`. If anything failed, one email goes to `MAIL_TO` listing each failure and its log, and the script exits `1`. No email is sent in a dry run.

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
