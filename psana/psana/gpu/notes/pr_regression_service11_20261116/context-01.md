# Independent compatibility review
Base: 980a7e7e3611c7561e4647af7d2fa0e126da8b94
Reviewed code: 0739a1ce92ae48af2d956af72a0cef88a1bf1ec5

User asks to adopt isEvent now if localized and inexpensive. Prior final review
flag NAT-009: service11 incorrectly counted as transition by benchmark observer.
Review the one predicate change and setup wiring in
psana/psana/gpu/scripts/performance/regression/native_worker.py, matching
observer tests in test_native.py, and description in NATIVE.md. Source authority:
psana/psana/psexp/transitionid.py and events.py. Prior finding details are in
psana/psana/gpu/notes/pr_regression_native_adapter_20261116/review-03.json.

Read scope.md, validation-01.json, measure_observer.py and timing.json in
psana/psana/gpu/notes/pr_regression_service11_20261116. Coordinator ran all45
offline tests (25adapter+20controller) successfully, diffcheckpassed. Existing
observer tests now use actual classifier loaded directly from source without
psana package/native initialization, include 11/12 mixed windows and drain,
service11-only first event window, transition-only first window, CPU events and
steps consumption, mismatch rejection, repeated drain and restoration of all
three hooks. Assess if sufficient, including any lost previous coverage.

Local timing: nine alternating pairs of100000stub service12 events, Python3.11
login34; median paired extra118.8ns/event, about1.19ms/10000. Not zero overhead;
not native throughput evidence. Full observer instrumentation overhead and all
existing native qualification gates remain pending. No performance assertions
in tests. No real dataset/MPI/GPU/Slurm run, profile or framework changes.
Skill guidance: source authoritative, keep per-event work small, preserve CPU
operation, no hardware throughput assertions in unit tests. Restrict to this
change; do not launch nested reviewers or execute tools beyond file reading.
Report denied reads. Resolve NAT-009 explicitly if supported by evidence.
