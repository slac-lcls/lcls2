# Historical psana2-GPU cold raw-staging control

Two same-node repeats reached5.040 and4.914GiB/s. [Combined report and settings](../jf_framework_cold_20261109/README.md) distinguish this configuration control from the matched raw/sim10 comparison. [Audit](validation.json) and [five tests](tests.log) pass. This launcher invokes the previously reviewed psana dense-staging bench, not standalone KvikIO. All10kevents/50kreads per run are through DataSource/MPI, with XTC parsing and dense gather. Native artifacts are on project scratch at the path recorded in the combined report.
