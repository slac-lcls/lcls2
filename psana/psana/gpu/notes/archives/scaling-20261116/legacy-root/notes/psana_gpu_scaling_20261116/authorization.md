# Session authorization

Mona explicitly approved the pending external Claude review request:

> Approve for review. Keep this in skill or for the session. If everything looks good, submit the job.

This answers the request to send this GPU-scaling plan, scripts, and validation
evidence, including scoped fixes in remaining review rounds, to external Claude.
It authorizes submission of the prepared qualification job after the review and
checks pass. Keep the normal pilot acceptance gate before any full screen.
This task/session authorization persists across turns and reviewer fix rounds;
do not ask for it again. It does not authorize unrelated source exports or
unbounded compute. No global skill or security rule is changed.

After qualification passed, Mona explicitly authorized the full scaling screen:

> Yes, let's do 100k.

This covers the prepared 12-case screen (three repetitions each at 1/2/3/4 GPUs,
100,000 total events per case), using the unchanged reviewed code and two-hour
one-node allocation. Job 61385304 was submitted under this authorization.
