CLEANUP-001 accepted/fixed: exact repr signal matching for all signals including
controller SIGTERM; step error must start TimeoutExpired(. Added SIGHUP rejects
SIGTERM and controller rejects SIGHUP regression tests.

CLEANUP-002 accepted/fixed: harness-cleanup.json records original in-flight error;
failed.json also records exception context.

CLEANUP-003 accepted/fixed: extracted pure validate_failure; rejection tests cover
wrong signal/type, unrelated cancelled ID, uncertain cleanup and timeout window.
Existing success-receipt checks remain in run_case and will be exercised on hardware.
No hardware execution yet.
