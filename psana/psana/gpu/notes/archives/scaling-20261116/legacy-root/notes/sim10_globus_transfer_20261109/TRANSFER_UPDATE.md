# Exact SDF Sim10 fixture transfer to Perlmutter

This update supersedes local run-442 fixture generation in
`notes/sim10_stream_length_20261109/TRANSFER.md`. The user authorized transferring
the accepted SDF 100k Sim10 fixture. Keep the controlled comparison and runtime,
scheduler, cache and network-counter adaptation requirements in that handoff.

Globus accepted task `a8426223-dca2-11f1-aa1a-0affd629e535` on
2026-11-10 at 01:04:21 UTC (November 9, 17:04:21 Pacific).
The task SUCCEEDED at 01:18:29 UTC (November 9, 17:18:29 Pacific): all 11 files,
335875429007 bytes, no faults or skipped files, checksum verification enabled.
All destination file sizes match, and independent SHA256 checks passed for all
five SMD files and generation.json. Bigdata SHA256 rereads and runtime benchmark
acceptance remain pending. See task-status.json and verification.json.
Timestamp preservation was enabled.

## Dataset and records

Destination on Perlmutter:
`/pscratch/sd/m/monarin/psana2-gpu/datasets/mfx101555026-r0010-sim10-100k-jf`

Transfer records on Perlmutter:
`/pscratch/sd/m/monarin/psana2-gpu/sim10-stream-length-20261109/transfer/globus-20261109`

The 11 selected files total 335875429007 bytes (312.808 GiB): five JF bigdata
streams s005..s009, five matching SMD files, and the unchanged generation.json.
They represent the exact JF subset used in the SDF 100k staging sweep.
Source names still identify mfx101555026 run 10. Use explicit dir= and JF-only
stream selection as appropriate; do not regenerate or rename the XTC payloads.

The original generation.json describes the complete ten-stream SDF fixture and
contains SDF absolute paths. Preserve it byte-for-byte for provenance. Use
transfer-manifest.json for this five-stream selection and path relocation.
Do not claim that all ten streams were transferred or that missing non-JF files
make this an invalid JF-only staging fixture.

Source collection: slac#s3df_globus5,
`31f8393f-79ef-4da4-a395-7127edd9b3a4`.
Verified source API path:
`/sdf/scratch/lcls/ds/prj/prjdatpsana/scratch/synthetic_mfx/sample10_100k/`.
The /sdf/data alias listing was delayed; use the verified resolved path.

Destination collection: NERSC DTN,
`9d6d994a-6d04-11e5-ba46-22000b92c6ec`.
Verified destination API path uses `/global/pscratch/sd/m/monarin/...`;
the login/compute path is `/pscratch/sd/m/monarin/...` as above.
Quota before transfer: 1160946987008 bytes used of 21990232555520 bytes.
Destination default Lustre stripe count is 1, stripe size 1048576 bytes.

## Continue after transfer

1. The transfer is complete. Inspect its saved receipt or this existing task;
   do not submit a duplicate. Load globus-tools for the command below:
   `globus task show a8426223-dca2-11f1-aa1a-0affd629e535`
2. Transfer acceptance is recorded in task-status.json and verification.json:
   SUCCEEDED, 11 files transferred, no skipped/failed files, expected byte count,
   and checksum verification. These checks are complete.
3. Compare destination file sizes and SHA256 values against the supplied
   transfer-manifest.json / SHA256SUMS before benchmark acceptance. Run the
   full-file verification in an appropriate compute allocation; the transfer
   receipt alone does not claim an independent SHA256 reread.
4. Revalidate Configure/SMD mapping, event count and the runtime's DataSource
   traversal on the relocated subset. Then port and run the small GPU pilots
   and the reviewed one-BD 1/2-stream x 10k/100k matrix on these same 100k files.
   Continue the implementation-review loop for code changes and native results.

This transfer launches no compute jobs and changes no calibration records.
Perlmutter Lustre remains a new site baseline, not a direct Weka reproduction.
