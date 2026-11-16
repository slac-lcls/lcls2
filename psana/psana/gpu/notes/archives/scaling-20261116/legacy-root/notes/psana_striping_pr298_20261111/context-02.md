Independent plan review round2/3; exact commit supplied by coordinator. Round1
passed a9bc787d95a3fcc0d323d671174a15b78e5d8834 with7nonblocking suggestions.
Read PLAN.md,scope.md,review-01.json,responses-01.md and the static PR test
inventory evidence/pr298-test-inventory.json. Recheck changes and regressions,
using the same local historical evidence listed in context-01.md as needed.
All required evidence is inside this checkout; no external scratch read needed.
No commands,edits,delegation or runtime experiments. This remains planning only.

Changes: require3complete blocks for ranking; separate full sizing case counts
against6hour cap but not rankings; fixed confirmation target4BD/2streams;
canonical24case list shuffled3times with one seeded RNG,one node/GPU/allocation;
explicit1/4peer discovery and6/24GiB aggregate admission/headroom; actual handle
fstat for bothKvikIO/DgramManager,all6counts get cheap untimed bindingpreflight;
immutable view once percount; six exact integration filenames and static unit
regression inventory atPR298head67c327a,with collected test IDs/counts required
at implementation review. No claim that static function inventory is actual
pytest collection. No implementation code or native benchmark has run.

Check no unresolved blocker and whether all7suggestions are addressed. Report
exactSHA,stable IDs,required read failures and evidence limitations explicitly.
