"""Keep historical GPU campaign snapshots out of routine pytest discovery.

Run their documented offline unittest/verifier commands explicitly instead.
Maintained GPU harness tests remain discoverable.
"""
collect_ignore = [
    "psana/psana/gpu/notes",
    "psana/psana/gpu/scripts/pr187_pr208",
    "psana/psana/gpu/scripts/pr299_perlmutter",
]
