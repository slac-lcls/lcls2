# `branch_out.sh`

Watches each hutch's production DAQ clones for local changes (files edited, added or deleted directly in the production directory) and records them on a branch on GitHub. That way every hutch-specific change is visible and reviewable, even though nobody committed it.

```
./branch_out.sh [--dry-run] <hutch_name> <root_dir> <branch_dir> <prefix>
```

| Argument | Meaning |
|---|---|
| `hutch_name` | Hutch, e.g. `xpp`. Used as the start of the branch name. |
| `root_dir` | Directory holding that hutch's production clones, e.g. `rel/xpp`. |
| `branch_dir` | A clone of `lcls2` used only to build and push these branches: `rel/branch_repo_lcls2/lcls2`. |
| `prefix` | `lcls`. Only directories whose names start with it are processed. |
| `--dry-run` | Report which files would change on which branches. Nothing in the branch repo is modified, and nothing is committed or pushed. |

Environment variable `LOG_ROOT` sets where failure reports are written (default `/sdf/group/lcls/ds/ana/sw/conda2/rel/cron_logs`).

Exit status is `0` if every matching clone synced, `1` if any failed.

---

## Overview

Each production clone gets its own branch, named `<hutch>-<clone directory>`:

```
rel/xpp/lcls2_092426   →   branch xpp-lcls2_092426
```

The directory name is included because a hutch usually has several clones (one per install), and each needs its own branch.

On every run, the branch is made to **match the clone exactly**: the commit the clone is on, plus all of its uncommitted changes. If a change is later undone in production, the next run undoes it on the branch too. The branch always shows what the clone looks like *now*, and its commit history shows how it got there.

The production clones are only **read**. All the work happens in the separate **branch repo**.

---

## Walkthrough

### 1. Setup and logging

```bash
set -e
LOG_ROOT="${LOG_ROOT:-/sdf/group/lcls/ds/ana/sw/conda2/rel/cron_logs}"
JUNK_PATTERN='(^|/)(\.[^/]*\.sw[a-p]|[^/]*~|\.#[^/]*|#[^/]*#|[^/]*\.out)$'
export GIT_OPTIONAL_LOCKS=0
```

- **`set -e`** stops the script on the first failing command, except in the per-clone section (see step 7).
- **`JUNK_PATTERN`** matches files that are never synced:
  - editor swap and backup files (vim's `.name.swp`, emacs's `name~`, `.#name` and `#name#`), which show up whenever someone has a file open in production
  - `*.out` files: saved command output. The first case found was `junk.out`, a 632 KB saved `git diff` in one of tmo's clones. Neither repo tracks any `*.out` file, so this can't hide a real change.
- **`GIT_OPTIONAL_LOCKS=0`**: even "read-only" git commands like `git status` quietly rewrite a repo's index file to cache file timestamps. This setting turns that off, so the script never writes anything into the production clones and never holds a lock that could get in the way of someone working there.

#### Failure reports

```bash
log_dir() {
    echo "${LOG_ROOT}/$(basename "${BRANCH_DIR:-unknown}")_branch"
}
```

`write_failure_report "<reason>"` writes a file to `<log_dir>/failed_runs/<timestamp>_<hutch>_<repo>_FAILED.log` (ending in `_FAILED_DRYRUN.log` during a dry run, so dry-run problems aren't mistaken for real failures) containing:
- the reason
- the inputs
- the clone being processed, its commit and branch name
- the list of changed files
- `git status` of both the clone and the branch repo

It's called when one clone fails (step 7), and by the `ERR` trap for unexpected failures in setup:

```bash
handle_error() {
    local exit_code=$?
    local line_number=$1
    write_failure_report "Unexpected error at line ${line_number} (exit code ${exit_code})"
    exit $exit_code
}
trap 'handle_error $LINENO' ERR
```

`log_dir` names the folder after the branch repo's directory, so for `rel/branch_repo_lcls2/lcls2` reports go to `<LOG_ROOT>/lcls2_branch/failed_runs/`. Earlier versions used `$4` inside the handler, but inside a function `$4` is the *function's* fourth argument, which was never passed. Every report therefore went to an `unknown_branch/` folder.

### 2. Arguments

The same `--dry-run` handling as the tag script: the flag can go anywhere, and the rest are the four positional arguments. The script checks that `root_dir` exists and `branch_dir` is a git repo.

### 3. Find the production clones

```bash
CANDIDATES=()
for candidate in "$ROOT_DIR"/*; do
    [ -d "$candidate" ] || continue
    [[ "$(basename "$candidate")" == ${PREFIX}* ]] || continue
    [ -d "$candidate/.git" ] || continue
    CANDIDATES+=("$candidate")
done
```

It collects every git repo in `root_dir` whose name starts with `prefix`. If there are none, it exits with an error. With the right `root_dir`, that points to a wrong path or prefix.

### 4. Trust exactly these repos (`safe.directory`)

```bash
SAFE_DIRS=("$BRANCH_DIR" "${CANDIDATES[@]}")
export GIT_CONFIG_COUNT=${#SAFE_DIRS[@]}
for i in "${!SAFE_DIRS[@]}"; do
    export "GIT_CONFIG_KEY_${i}=safe.directory"
    export "GIT_CONFIG_VALUE_${i}=$(cd "${SAFE_DIRS[$i]}" && pwd -P)"
done
```

Git refuses to work in a repo owned by a different user ("dubious ownership"). This is a protection against someone else's repo running code as you. `safe.directory` lists repos to trust anyway.

`GIT_CONFIG_COUNT` / `GIT_CONFIG_KEY_n` / `GIT_CONFIG_VALUE_n` set git config for **this process only**. The script trusts exactly the branch repo and the clones it's about to read, and nothing else. `pwd -P` gives each repo's real path, with symlinks resolved, which is what git compares against.

Earlier versions ran `git config --global --add safe.directory '*'`, which permanently turned the check off for every repo `psrel` touches.

### 5. Prepare the branch repo (once)

```bash
cd "$BRANCH_DIR"
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${RED}Error: Base repo is not clean. Aborting.${NC}"
    write_failure_report "Branch repo is not clean"
    exit 1
fi

git fetch --prune origin
git checkout -q master
if ! git merge --ff-only -q origin/master; then
    echo -e "${YELLOW}Warning: could not fast-forward master to origin/master; continuing${NC}"
fi
```

- **Clean check.** The branch repo should never have leftover changes. If it does, the script stops rather than stash them, because a stash would easily be forgotten. It also means that later, after a failure, the script can safely reset the repo (step 7).
- **`git fetch --prune origin`** gets GitHub's current branches. `--prune` removes local records of branches that were deleted on GitHub, so "does this branch exist on GitHub?" (step 6c) is answered correctly.
- **`master`** is just where the repo sits between syncs. `--ff-only` updates it without ever creating a merge commit. If that fails, it's only a warning, because nothing depends on it.

This happens **once per run**. Earlier versions did the checkout and `git pull` again for every clone. That step doesn't depend on the clone, so one failed pull (as on 2026-09-13) ended the whole run.

In a **dry run**, none of this happens (no fetch, no checkout). The dry run works from the branch repo's last fetch.

### 6. Sync one clone: `sync_repo`

#### a. Read the clone's commit

```bash
GIT_HASH=$(git -C "$MONITOR_REPO" rev-parse HEAD)
BRANCH_NAME="${HUTCH_NAME}-${REPO_NAME}"
```

`HEAD` is the commit the clone currently has checked out. `git -C <dir>` runs git in that directory without `cd`-ing into it.

#### b. List the clone's changes, without changing anything

```bash
git -C "$MONITOR_REPO" diff -z --name-only --no-renames HEAD > "$TMP_FILE"
mapfile -d '' tracked < "$TMP_FILE"

git -C "$MONITOR_REPO" ls-files -z --others --exclude-standard > "$TMP_FILE"
mapfile -d '' untracked < "$TMP_FILE"
```

- `git diff --name-only HEAD` lists **tracked** files that differ from the commit: modified or deleted, whether staged or not.
- `git ls-files --others --exclude-standard` lists **new** files git doesn't track yet, skipping anything matched by `.gitignore` (so `build/`, `install/`, `*.pyc` and so on are never picked up).
- Junk files matching `JUNK_PATTERN` are dropped. The rest go into `want` (a set of wanted files) and `CHANGED_FILES`.

Details:
- **`-z` and `mapfile -d ''`** separate file names with a null character instead of a newline or space, so names containing spaces are handled correctly. The list goes through a temp file because bash variables can't hold null characters.
- **`--no-renames`** makes a renamed file show up as "old name deleted" plus "new name added". Otherwise only the new name is listed and the old file would never be removed from the branch.

**Why not `git add --all`:** earlier versions ran `git add --all` in the production clone so new files would show up in `git diff`. That *staged* every change in the live DAQ repo. Anyone running `git status` or `git commit` there later found changes staged that they didn't make, including swap files. Reading the two lists separately gives the same information without touching the clone.

#### c. Choose what the branch starts from

```bash
if ! git rev-parse --verify --quiet "${GIT_HASH}^{commit}" >/dev/null; then
    FAIL_STEP="commit $GIT_HASH not found in branch repo (does the production clone have local commits?)"
    return 1
fi

if git rev-parse --verify --quiet "refs/remotes/origin/${BRANCH_NAME}" >/dev/null; then
    base_ref="origin/${BRANCH_NAME}"          # existing branch on origin
elif git rev-parse --verify --quiet "refs/heads/${BRANCH_NAME}" >/dev/null; then
    base_ref="refs/heads/${BRANCH_NAME}"      # local branch not yet on origin
else
    base_ref="$GIT_HASH"                      # new branch from clone commit
fi
```

First, the clone's commit must exist in the branch repo. If someone committed directly in production and never pushed, it won't, and that clone fails with a clear message.

Then the starting point, in order of preference:
1. **The branch on GitHub**, if it exists. GitHub is the source of truth, so a re-cloned or out-of-date branch repo still continues the existing branch. Earlier versions only looked at local branches. After a re-clone they'd start a fresh branch from the commit, and the push would be rejected because GitHub's branch had different history.
2. **A local branch that isn't on GitHub**: left by a run whose push failed. It gets pushed this time.
3. **The clone's commit**: a brand-new branch.

#### d. Decide what has to change on the branch

```bash
git diff -z --name-only --no-renames "$GIT_HASH" "$base_ref" > "$TMP_FILE"
mapfile -d '' touched < "$TMP_FILE"
```

`touched` lists every file where the branch currently differs from the clone's commit, i.e. every change the branch currently carries. The script then looks at each file in the clone's changes **plus** `touched`, and gives each one an action:

| File is… | Action | What happens |
|---|---|---|
| changed in the clone, and exists there | `copy` | copy it from the clone |
| changed in the clone, and was deleted there | `delete` | delete it on the branch |
| **not** changed in the clone any more, but still changed on the branch | `restore` | reset it to the clone's commit (or remove it, if it isn't in that commit) |

The `restore` row is what makes the branch follow production when a change is **undone**. Earlier versions only copied the currently-changed files on top of the old branch, so a reverted change stayed on the branch forever. It also cleans up junk files that older versions pushed (for example, editor swap files).
One real case: the old script pushed `.setup_env_newtest.sh.swp` to `xpp-lcls2_060226`, and the first run of this version removes it.

Files that already match are dropped from the plan, by comparing git's content hashes:

```bash
worktree_blob() {           # hash of the file in the clone
    ...
    git hash-object -- "$path"
}
commit_blob() {             # hash of the file at a commit, empty if absent
    git -C "$BRANCH_DIR" rev-parse --verify --quiet "$1:$2" 2>/dev/null || true
}
```

`git hash-object` computes the hash git *would* give the file, without storing anything. If the clone's file and the branch's file have the same hash, their contents are identical. For symlinks, git stores the link's target path, so `worktree_blob` hashes that instead of the file the link points to.

The whole plan is worked out **without modifying anything**. That's what makes `--dry-run` accurate: it prints the plan and stops.

#### e. Apply, commit and push

This only happens if the plan is not empty, or if the branch exists only locally and needs pushing.

```bash
run git checkout -q -B "$BRANCH_NAME" "$base_ref" || return 1
```

`-B` creates the local branch, or resets it if it already exists, at the chosen starting point.

Then each action:
- **copy**: `rm -f` the old file, `cp -pP` the clone's version in, then `git add -f`.
  - `-p` keeps permissions, so new scripts stay executable. Earlier versions lost the executable bit.
  - `-P` copies a symlink as a symlink.
  - `add -f` adds the file even if the branch repo's `.gitignore` would ignore it. It's in the plan because it's a real change in production.
- **delete**: `git rm`.
- **restore**: `git checkout <clone commit> -- <file>`, or `git rm` if the file isn't in that commit.

```bash
run git commit -q -m "Sync from ${PROJECT} (${BRANCH_NAME})
Source: $MONITOR_REPO
Source commit: $GIT_HASH
Files synced:
copy: a.txt
restore: b.txt ..."
```

The commit message says where the change came from, and lists every file and action. `PROJECT` is the branch repo's directory name (`lcls2`); earlier versions had the project name hardcoded.

```bash
if git rev-parse --verify --quiet "refs/remotes/origin/${BRANCH_NAME}" >/dev/null; then
    ahead=$(git rev-list --count "origin/${BRANCH_NAME}..${BRANCH_NAME}")
else
    ahead=1
fi
if [ "$ahead" -gt 0 ]; then
    run git push -q -u origin "$BRANCH_NAME" || return 1
fi
run git checkout -q master || return 1
```

It pushes **whenever the branch has commits GitHub doesn't**, not only right after a new commit. Earlier versions only pushed after a new commit. If a push failed, the next run found nothing new to commit and never tried again. Finally, it goes back to `master`.

### 7. The main loop: one failure doesn't stop the rest

```bash
for candidate in "${CANDIDATES[@]}"; do
    if sync_repo "$candidate"; then
        SYNCED=$((SYNCED + 1))
    else
        FAILED+=("${REPO_NAME}: ${FAIL_STEP}")
        write_failure_report "$FAIL_STEP"
        if ! $DRY_RUN && ! reset_branch_repo; then
            ... exit 1
        fi
    fi
done
```

If one clone fails, the script records it, writes a failure report, puts the branch repo back to a clean `master`, and carries on with the next clone:

```bash
reset_branch_repo() {
    git -C "$BRANCH_DIR" reset -q --hard &&
        git -C "$BRANCH_DIR" clean -fdq &&
        git -C "$BRANCH_DIR" checkout -q master
}
```

`reset --hard` and `clean -fd` throw away any half-applied changes. That's safe only because step 5 verified the repo was clean before any sync started, so anything there now came from this run.

**A bash detail that matters here:** `set -e` is *switched off* inside a function called as the condition of an `if` (`if sync_repo ...`). So `sync_repo` can't rely on `set -e` to stop at a failing command. Every command that can fail is checked explicitly, mostly through the small `run` helper. It records the failed command **and git's reason** in `FAIL_STEP`, and still prints the full error output to the log:

```bash
run() {
    if "$@" 2> "$ERR_FILE"; then
        cat "$ERR_FILE" >&2
        return 0
    fi
    cat "$ERR_FILE" >&2
    FAIL_STEP="failed: $*$(err_suffix)"
    return 1
}
```

`err_suffix` picks git's actual reason out of its error output. Git often ends with a generic line: `error: failed to push some refs`, or for SSH problems `...and the repository exists.` So it quotes the last two lines that carry a real reason (`fatal:`, `error:`, `remote:`, `! [rejected] …`, `Permission denied`), falling back to the last non-empty line. For example:

```
failed: git push -q -u origin xpp-lcls2_060226 (git@github.com: Permission denied (publickey).; fatal: Could not read from remote repository.)
failed: git push -q -u origin xpp-lcls2_060226 (! [rejected] xpp-lcls2_060226 -> xpp-lcls2_060226 (non-fast-forward))
```

The git calls that read the clone (`rev-parse`, `diff`, `ls-files`) add the same suffix to their failure reasons. The reason appears in the error line, in the summary, in the failure report, and (through `run_monitor.sh`) in the failure email.

### 8. Summary and exit status

```
=== Summary ===
Repos processed: 19
Succeeded:       19
Failed:          0
```

It exits `1` if any clone failed.

---

## Things to know

- **New untracked files are included**: anything not matched by `.gitignore` or `JUNK_PATTERN`. To sync only tracked files, drop the `ls-files --others` step in 6b.
- **Clean clones get no branch.** A clone with no changes never gets a branch created.
- **The branch repo is disposable.** Delete and re-clone it (SSH remote) at any time. The next run continues from GitHub.
- **If a clone's `HEAD` moves** (someone pulls in production), the next sync brings the branch's files in line with the new commit, and the upstream differences show up as one commit on the branch.
