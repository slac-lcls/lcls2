# Response to review round 1

Reviewed commit c5537663b024386b136e45594a57fe2d0d492d1c; verdict changes_requested.
Reviewer process completed successfully, with no permission denials. This is a
document review, not native results acceptance.

| ID | Response | Evidence |
|---|---|---|
| SEL-R1-01, blocking | Accepted and corrected 18.4% to **16.4%**. Coordinator re-rendered PDF page 3 at 150 dpi and visually checked the Weka/one-GPU cell; the adjacent cells are 4.2% and 13.2%. | README precise page/panel/cell citation; evidence/repeat-spread.png supplied for direct image reading; validation.json records visual transcription and image hash. Original PDF unchanged. |
| SEL-R1-02, nonblocking | Accepted. Added the reversed ordering at batch100, untested four-BD/sixteen-worker point, and resource-based reason for choosing four workers. | README evidence/decision section; source report and heatmap. |
| SEL-R1-03, nonblocking | Accepted. Independently regenerated all 18 schedule (case,block,count) tuples from the stated seed and compared against the saved CSV. | validation.json records seed, interpreter version, exact order, CSV SHA256 and successful equality. The CSV has not changed. |

All fixes await independent re-review. No jobs, harness changes, runtime upgrade,
or change to the selected configuration is involved.
