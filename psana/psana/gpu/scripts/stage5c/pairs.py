"""Acceptance for matched numerical work; no throughput thresholds."""


def validate_pair(event, batch):
    assert event['variant']=='event_loop' and batch['variant']=='batched_task'
    for key in ('events','ngpus','nbds','batch_size','depth','cache','bulk','gpu_buses',
                'pipeline_budget_gb','diagnostic','output_sha256'):
        assert event[key]==batch[key], ('unmatched pair',key,event[key],batch[key])
    assert event['loop_s']>0 and batch['loop_s']>0
    return dict(delta_s=batch['loop_s']-event['loop_s'],
                percent=100*(batch['loop_s']/event['loop_s']-1))
