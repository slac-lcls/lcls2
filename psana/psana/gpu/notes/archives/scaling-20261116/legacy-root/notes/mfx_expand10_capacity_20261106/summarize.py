"""Verify campaign reports/source/retained hashes and produce compact Git evidence."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(4<<20),b''):digest.update(data)
    return digest.hexdigest()


def main(root, output, recovery_root=None):
    complete=json.loads(((recovery_root or root)/'completion.json').read_text())
    assert complete['status']=='pass' and complete['bins']==([200] if recovery_root else [30,100,200])
    if recovery_root:
        launcher=json.loads((recovery_root/'launcher.json').read_text())
        assert sha(launcher['path'])==launcher['sha256']
    controlled=json.loads((root/'controlled-tests.json').read_text());assert controlled['status']=='pass'
    rows=[];seed_hashes=set()
    for nbins in (30,100,200):
        case_root=recovery_root if recovery_root and nbins==200 else root
        path=case_root/f'bins-{nbins}/report.json';r=json.loads(path.read_text())
        assert r['status']=='pass' and r['events']==1000 and r['export_count']==1
        assert r['verification']['all_counts_exact'] and r['verification']['all_pixels_within_bound']
        assert r['verification']['compared_pixels']==nbins*32*512*1024
        assert r['persistent_bytes_total']==2*nbins*(32*512*1024*4+8)
        assert r['export']['reusable_host_array_bytes']==32*512*1024*12+nbins*8
        assert r['temporary_cube_removed']
        seed_hashes.add(r['seed_cache_manifest']['seed_cache_sha256'])
        for code,digest in r['code_sha256'].items():assert sha(code)==digest,code
        for code in r['runtime_files'].values():assert sha(code['path'])==code['sha256'],code
        kept=[]
        for b in r['verification']['retained_bins']:
            f=path.parent/'retained-bins'/f'bin-{b:05d}.npy'
            assert sha(f)==r['verification']['bins'][b]['file_sha256']
            kept.append(dict(path=str(f),sha256=sha(f),bytes=f.stat().st_size))
        row={k:r[k] for k in ('nbins','active_bins','events','job','runtime','runtime_files','calibration','generation_sha256',
                              'admission','persistent_bytes_total','gpu_phase_memory','bank_event_counts','iteration_wall_seconds',
                              'kernel_compile_seconds','datasource_setup_seconds','read_stats','export_count','code_sha256')}
        row.update(report=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size),
                   cuda_stages=r['cuda_timing']['stages'],export=r['export'],host_lifetime_peak=r['memory']['peaks']['VmHWM_bytes'],
                   verification={k:v for k,v in r['verification'].items() if k!='bins'},
                   max_abs_error=max(b['max_abs_error'] for b in r['verification']['bins']),
                   max_bound_ratio=max(b['max_bound_ratio'] for b in r['verification']['bins']),
                   retained_bins=kept)
        rows.append(row)
    assert len(seed_hashes)==1,'Cases used different CPU seed caches'
    summary=dict(status='pass',artifact_root=str(root),controlled_gpu_tests=controlled,results=rows,
                 all_recorded_code_hashes_match=True,all_retained_artifact_hashes_match=True,
                 common_seed_cache_sha256=next(iter(seed_hashes)),
                 temporary_workspace_removed=complete['local_workspace_removed'],
                 recovery_root=str(recovery_root) if recovery_root else None,
                 interrupted_job='43533593' if recovery_root else None)
    output.write_text(json.dumps(summary,indent=2)+'\n')
    for r in rows:
        print(json.dumps(dict(bins=r['nbins'],device_peak_gib=r['gpu_phase_memory']['device_used_bytes']/(1<<30),
                              pool_peak_gib=r['gpu_phase_memory']['cupy_pool_total_bytes']/(1<<30),
                              host_gpu_gib=r['gpu_phase_memory']['VmHWM_bytes']/(1<<30),
                              host_lifetime_gib=r['host_lifetime_peak']/(1<<30),iteration_s=r['iteration_wall_seconds'],
                              export_s=r['export']['elapsed_seconds'],max_bound_ratio=r['max_bound_ratio'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--recovery-root',type=Path)
    args=p.parse_args();main(args.root,args.output,args.recovery_root)
