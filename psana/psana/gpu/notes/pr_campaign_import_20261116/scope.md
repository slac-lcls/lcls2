# PR study import scope

Import the two pinned PR187/208 and PR299 documentary packages into the clean
results/psana2-gpu-scaling branch. Preserve campaign scripts and evidence bytes,
retain PR299's verifier-sensitive paths, reconcile current navigation, and
isolate historical sources from pytest/lint/formatting and package installation.
Base: 12366915ee2e7991176c19b03d4a24134e76c7f3.
Acceptance: closed source-delta inventory; exact imported hashes; original
PR299 verifier and tampering tests; PR187's 47 snapshot hashes and six CPU
tests; useful report links; runtime/examples unchanged; collection and package
exclusion checks; independent Claude review on an exact clean commit.
Excludes scaling migration, shared regression runner, dataset retrieval,
hardware jobs, remote publishing, PR creation and source-branch retirement.
Default review limit: three rounds, 15 minutes per reviewer invocation.
