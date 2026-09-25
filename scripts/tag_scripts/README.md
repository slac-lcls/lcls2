# Tag and Branch Scripts

Scripts that keep a record on GitHub of which version of `lcls2` is deployed in each hutch's production DAQ directories, and of any local changes made there.

Detailed docs are in [`docs/`](docs/):
- [`docs/single_push_collective_tag.md`](docs/single_push_collective_tag.md)
- [`docs/branch_out.md`](docs/branch_out.md)
- [`docs/run_monitor.md`](docs/run_monitor.md)
- [`docs/deployment.md`](docs/deployment.md): where things live, cron, and adding a hutch

All three scripts accept `--dry-run`, which reports what would happen without creating, committing or pushing anything.

---

## `single_push_collective_tag.sh`

### Purpose

Tags the commit each production `lcls2` clone in a hutch was installed from, as `<hutch>-<YYYYMMDD of install>` (e.g. `xpp-20260908`). This gives accurate bookkeeping of which version was deployed in each hutch on each date.

### Usage

```
./single_push_collective_tag.sh [--dry-run] <hutch_name> <root_dir> <tag_repo_path> <prefix>
```

- **`hutch_name`**: the hutch, e.g. `xpp`, `tmo`. Used in the tag name.
- **`root_dir`**: the directory holding the hutch's production clones, e.g. `/sdf/group/lcls/ds/ana/sw/conda2/rel/xpp`.
- **`tag_repo_path`**: the repo used only for pushing tags, `/sdf/group/lcls/ds/ana/sw/conda2/rel/tag_repo_lcls2/lcls2`.
- **`prefix`**: `lcls`. Only clones whose directory names start with it are tagged.

---

## `branch_out.sh`

### Purpose

Monitors the production `lcls2` clones in a hutch. When a clone has local changes, it records them on a branch named `<hutch>-<clone directory>` (e.g. `xpp-lcls2_092426`) and pushes it to the `lcls2` repo. The branch always matches what the clone currently looks like.

### Usage

```
./branch_out.sh [--dry-run] <hutch_name> <root_dir> <branch_dir> <prefix>
```

- **`hutch_name`**: the hutch, e.g. `xpp`, `tmo`. Used in the branch name.
- **`root_dir`**: the directory holding the hutch's production clones.
- **`branch_dir`**: the repo used to build and push the branches, `/sdf/group/lcls/ds/ana/sw/conda2/rel/branch_repo_lcls2/lcls2`.
- **`prefix`**: `lcls`. Only clones whose directory names start with it are processed.

---

## `run_monitor.sh`

### Purpose

What cron runs. Runs the branch or tag job for `lcls2` for every monitored hutch, one at a time. It writes a dated log per hutch and sends one email if anything failed.

### Usage

```
./run_monitor.sh [--dry-run] <branch|tag> [hutch ...]
```

- **`branch`** or **`tag`**: which job to run.
- **`hutch ...`** (optional): hutches to run for. Default is the `HUTCHES` list at the top of the script.

Example: `./run_monitor.sh --dry-run branch tmo`
