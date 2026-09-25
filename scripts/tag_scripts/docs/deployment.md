# Deployment

Where everything for `lcls2` monitoring lives, how cron runs it, and how to set it up, move to it from the old setup, and add hutches.

`rel` below means `/sdf/group/lcls/ds/ana/sw/conda2/rel`.

---

## Where things are

```
rel/
  <hutch>/                          production DAQ clones, one directory per hutch
    lcls2_MMDDYY/                       created by install_release_daq.sh
  branch_repo_lcls2/lcls2/              shared branch repo
  tag_repo_lcls2/lcls2/                 shared tag repo
  monitor_scripts/lcls2/              checkout of lcls2 that cron runs the scripts from
    scripts/tag_scripts/
      run_monitor.sh  branch_out.sh  single_push_collective_tag.sh
  cron_logs/                        logs (see run_monitor.md)
```

Hutches: `mfx`, `rix`, `tmo`, `txi`, `ued`, `xpp`.

### Production clones

Each hutch installs its own DAQ by running `install_release_daq.sh` (as `psrel`) inside `rel/<hutch>/`. It clones `lcls2` (along with the other DAQ repos) into `lcls2_<MMDDYY>` and builds it. Hutches keep their own clones because each hutch sometimes needs changes specific to it. Capturing those changes is what `branch_out.sh` is for.

The scripts only look at the **top level** of `rel/<hutch>/`, and only at directories whose names start with `lcls`. Keep anything else (scripts checkouts, tools) out of the hutch directories, or give it a name that doesn't start with `lcls`.

### The shared branch and tag repos

`lcls2` has one branch repo and one tag repo, **shared by all hutches**:
- **Tag and branch names always start with the hutch** (`xpp-20260908`, `xpp-lcls2_092426`), so hutches never collide.
- **`run_monitor.sh` runs hutches one at a time under a lock**, so they never use a repo at the same time.
- **GitHub is the source of truth.** The scripts check GitHub for existing tags and branches, so neither repo needs to be kept in sync with anything, and either can be deleted and re-cloned.

Requirements:
- **SSH remote** (`git@github.com:slac-lcls/lcls2.git`). `psrel` can only push over SSH.
- **Clean working trees.** Both scripts stop if their repo has uncommitted changes. Don't edit files in them.
- The branch repo needs a `master` branch.

To re-create one (as `psrel`):
```
rm -rf rel/branch_repo_lcls2/lcls2
git clone git@github.com:slac-lcls/lcls2.git rel/branch_repo_lcls2/lcls2
```

### The scripts checkout

`rel/monitor_scripts/lcls2` is a clone of `lcls2` used **only** to run these scripts. cron calls `run_monitor.sh` from it, and `run_monitor.sh` calls the other two scripts from the same folder, so all three are always the same version.

Keep it outside the hutch directories. To update the scripts, merge the change into the branch this checkout follows, then run `git pull` in it.

`run_monitor.sh` has `PROJECT=lcls2` and `PREFIX=lcls` at the top; that is what makes this copy handle `lcls2`. The same scripts are kept in other DAQ repos with their own `PROJECT`/`PREFIX`. The job scripts `branch_out.sh` and `single_push_collective_tag.sh` are identical in every copy. When you fix something in them, make the same change in every copy.

---

## Account and host

- cron runs on **`sdfcron001`** as **`psrel`**. Log in as `psrel@sdfcron001` to see or edit the crontab.
- `psrel` is used (not a personal account) so the whole dev team can maintain the jobs, and because `psrel` owns the production clones and has push access to `lcls2` over SSH.

---

## Cron

```cron
# ---- lcls2 BRANCH (daily) ----
0 2 * * * /sdf/group/lcls/ds/ana/sw/conda2/rel/monitor_scripts/lcls2/scripts/tag_scripts/run_monitor.sh branch >> /sdf/group/lcls/ds/ana/sw/conda2/rel/cron_logs/run_monitor_lcls2.log 2>&1

# ---- lcls2 TAG (weekly, Sunday) ----
0 4 * * 0 /sdf/group/lcls/ds/ana/sw/conda2/rel/monitor_scripts/lcls2/scripts/tag_scripts/run_monitor.sh tag >> /sdf/group/lcls/ds/ana/sw/conda2/rel/cron_logs/run_monitor_lcls2.log 2>&1
```

- **Branch runs daily at 02:00. Tag runs Sundays at 04:00**, later so it doesn't start while a branch job is running.
- **Overlaps are handled by the lock.** Other projects' monitor jobs can be scheduled at the same times: they all share the lock in `rel/cron_logs/`, so whichever starts second waits for the first to finish.
- **The tag job must keep running at least weekly.** It reads each clone's install date from git's reflog, which git deletes after 90 days. A clone not tagged within 90 days of install can't be tagged afterwards.
- **These two lines cover every hutch.** The hutch list is `HUTCHES` in `run_monitor.sh`.
- **`>>` appends** the wrapper's short summary to `run_monitor_lcls2.log`. Each hutch's full output goes to its own dated log.
- **Failure email** is sent by `run_monitor.sh` itself (`MAIL_TO`), so the crontab doesn't need `|| mail`.
- **`%` is special in crontab** and must be written `\%` if you ever add one to a line.

---

## One-time setup (as `psrel` on `sdfcron001`)

These are per account, not per project. If they were already done while setting up another project's monitoring, skip them.

1. **Git identity**, so branch commits aren't attributed to a name git makes up:
   ```
   git config --global user.name "<name>"
   git config --global user.email "<email>"
   ```
2. **Remove the global `safe.directory '*'`** added by earlier versions of `branch_out.sh`. The script now trusts only the repos it uses, for its own process.
   ```
   git config --global --get-all safe.directory            # see what's there
   git config --global --unset-all safe.directory '^\*$'   # removes only the '*' entry
   ```
3. **Check SSH push access:** `ssh -T git@github.com` should greet the account.

---

## Setting up (and moving from the old setup)

The old setup ran two crontab lines for `lcls2`, for `xpp` only, from two different checkouts on `features/scripts`: `rel/xpp/branch_lcls2` for the branch job, and the tag repo `rel/tag_repo_lcls2/lcls2` itself for the tag job. The lines are the ones whose arguments end in `rel/branch_repo_lcls2/lcls2 lcls` and `rel/tag_repo_lcls2/lcls2 lcls`. Logs went to fixed files that were overwritten every run.

All as `psrel` on `sdfcron001`:

1. **Check the shared repos.** They should already exist, with SSH remotes and clean working trees:
   ```
   R=/sdf/group/lcls/ds/ana/sw/conda2/rel
   for r in branch_repo_lcls2/lcls2 tag_repo_lcls2/lcls2; do
     echo "$r: $(git -C $R/$r remote get-url origin)  $(git -C $R/$r status --porcelain | wc -l) changed files"
   done
   ```
   Each line should show `git@github.com:slac-lcls/lcls2.git` and `0 changed files`. If a repo is missing, clone it (see "The shared branch and tag repos").
2. **Create the scripts checkout**, on the branch that has these scripts (`master` once merged):
   ```
   mkdir -p $R/monitor_scripts
   git clone -b <branch> git@github.com:slac-lcls/lcls2.git $R/monitor_scripts/lcls2
   ls $R/monitor_scripts/lcls2/scripts/tag_scripts
   ```
3. **Do the one-time setup** above, if not already done.
4. **Dry run**, and read the logs it points to:
   ```
   S=$R/monitor_scripts/lcls2/scripts/tag_scripts
   $S/run_monitor.sh --dry-run branch
   $S/run_monitor.sh --dry-run tag
   ```
   When tested on 2026-09-25, a dry run for `xpp` showed:
   - **tag:** two new tags, `xpp-20260918` and `xpp-20260924` (clones installed since the last tag run on 2026-09-13). The other 17 were already tagged.
   - **branch:** 19/19 clones ok. A single change: removing a vim swap file (`.setup_env_newtest.sh.swp`) that the old script had pushed to `xpp-lcls2_060226`. Every other branch already matched its clone.
5. **Back up the crontab, then swap the lines.** Replace the two old `lcls2` lines with the two lines under "Cron" above:
   ```
   crontab -l > ~/crontab.backup.$(date +%Y%m%d)
   crontab -e
   ```
6. **Check the first runs**: `rel/cron_logs/run_monitor_lcls2.log`, then the per-hutch logs under `rel/cron_logs/xpp/lcls2_branch/` and `…/lcls2_tag/`.
7. **Clean up** once the new setup has run for a while. None of this is needed for it to work:
   - Old log folders at the top of `rel/cron_logs/`: `lcls2_branch/`, `lcls2_tag/`, and `unknown_branch/` (failure reports from the old script). New logs are under `rel/cron_logs/<hutch>/`.
   - `rel/xpp/branch_lcls2`, the old scripts checkout inside a hutch directory. **Only remove it once `crontab -l` shows no line using it**: the old setup ran other jobs from it too.
   - The tag repo `rel/tag_repo_lcls2/lcls2` is checked out on `features/scripts`, because the old tag lines ran the script from inside it. Once no crontab line uses it, switch it back: `git -C $R/tag_repo_lcls2/lcls2 checkout master`.
   - **Staged changes in the xpp production clones**, left behind by the old script's `git add --all` (e.g. `rel/xpp/lcls2_060226`). They're harmless, and the new script doesn't care. Clearing them (`git reset` in each clone, which only unstages and doesn't change any file) touches production, so agree it with the hutch first.

---

## Adding a hutch

1. Make sure the hutch's clones are in `rel/<hutch>/` (installed with `install_release_daq.sh`).
2. **Dry run for just that hutch** and read the logs:
   ```
   $S/run_monitor.sh --dry-run branch <hutch>
   $S/run_monitor.sh --dry-run tag <hutch>
   ```
   The first real run creates a branch for every clone with local changes, and a tag for every clone whose reflog still has its clone entry (normally clones up to about 90 days old). Older clones show up as skipped.
3. **Add the hutch to `HUTCHES`** in `run_monitor.sh`, commit and push it, and `git pull` in `rel/monitor_scripts/lcls2`.

The crontab doesn't change.

---

## When something fails

You get an email listing each failed hutch and its log file.

| Message in the log | Meaning / what to do |
|---|---|
| `Base repo is not clean` / `Tag repo has uncommitted changes` | Someone edited files in a shared repo. Look at `git status` there, then clean it up or re-clone it. |
| `commit … not found in branch repo (does the production clone have local commits?)` | Someone committed directly in that production clone and didn't push. Only that clone is affected; the others still sync. |
| `Commit … not found in tag repo` (listed under Skipped) | The clone was made from a fork, or from a branch deleted on GitHub. It can't be tagged. |
| `no clone entry in reflog` (Skipped) | The clone's reflog has expired (older than 90 days, or copied rather than cloned). It can't be tagged. |
| `failed: git push …` / `Tag push failed` | Usually network or GitHub access. Nothing is lost; the next run pushes it. |
| `an earlier run still held the lock` | A run (this project's or another project's) has been going for over 2 hours, or is stuck. Check for a hung `run_monitor.sh` process on `sdfcron001`. |

Branch failures also write a detailed report (inputs, file list, `git status` of both repos) to `rel/cron_logs/<hutch>/lcls2_branch/failed_runs/`.
