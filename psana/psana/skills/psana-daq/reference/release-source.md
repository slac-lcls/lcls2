# Release-Source Navigation (cross-cutting)

When any skill needs to read source ground truth — to look up a message,
verify a timeout, or check a guard — use this technique to locate the exact
source tree the running DAQ was built from. `psana-daq-logs` and
`psana-daq-control` both cite this section; cite it in turn rather than
re-deriving the technique.

**How to find the source:**

1. Read `# TESTRELDIR:` from the log file header. `TESTRELDIR` points at
   `<root>/install` (compiled output — `bin/` and `lib/` only). Source is
   at the **parent**: strip the trailing `/install`.
2. The source root contains e.g. `psdaq/psdaq/control/control.py` —
   confirmed readable.

**Identifying the release**: for conda releases the `lcls2_<MMDDYY>`
component of the `TESTRELDIR` path is the version marker. Don't try to
parse a universal pattern — the path may be a developer home directory
(e.g. `/sdf/home/w/weaver/lcls2/install`), not a conda release.

**`git` may be unusable in these trees**: release trees are frequently
owned by a build account distinct from the reading user, and git refuses
with "dubious ownership" in that case. Plain file reads and greps work
regardless. If `git` is needed and refuses, a dirty build can also differ
from its nominal commit — note that possibility rather than assuming the
commit matches the checkout exactly.

**`GIT_DESCRIBE`** appears in session headers once `daqlog_header.py`'s fix
has propagated to the release in use; check whether it's present rather
than assuming either way — the `lcls2_<MMDDYY>` path component is the
reliable fallback marker regardless.

**Multiple releases coexist** — e.g. three different ones in one month at
one hutch. Line numbers shift between releases by inconsistent amounts (a
message at `:2255` in one release can be at `:2220` in another). A citation
valid in one release is wrong in another and yields a plausible-looking but
incorrect answer — always state which tree was used.

**If unsure which tree to read, ask.** Ask when: multiple `TESTRELDIR`
values are in play, the path doesn't match the conda convention, the tree
is unreadable, or the question spans more than one release.
