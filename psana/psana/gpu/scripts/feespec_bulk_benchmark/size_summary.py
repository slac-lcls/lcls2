"""Two-round 4 MiB bulk target/task comparison at pool depth one."""
from depth_summary import summarize as summarize_common


def matrix():
    return ([('size4', v, 'warm', 1, 'pipeline') for v in ('E-off', 'E-on')]
            + [('size4', v, cache, r, 'control') for r in (1, 2)
               for cache in (('cold', 'warm') if r == 1 else ('warm', 'cold'))
               for v in (('E-off', 'E-on') if r == 1 else ('E-on', 'E-off'))]
            + [('size4', v, 'cold', r, 'trace') for r in (1, 2)
               for v in (('E-off', 'E-on') if r == 1 else ('E-on', 'E-off'))])


def summarize(root, results, *, complete=False):
    return summarize_common(root, results, complete=complete, builds=('size4',),
        title='4 MiB bulk target and KvikIO tasks; pool depth 1', task_mib=4,
        schedule=matrix())
