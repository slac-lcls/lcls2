"""Derived comparisons and inventory checks; preserve individual native repeats."""
import json,statistics,sys
from pathlib import Path
from support import require,write,sha

def summarize(root):
    a=json.loads((root/'audit.json').read_text());p=json.loads((root/'provenance.json').read_text())
    def pci(value):
        domain,bus,slot=value.strip().split(':');device,function=slot.split('.')
        return tuple(int(x,16) for x in (domain,bus,device,function))
    inventory=[list(map(str.strip,line.split(','))) for line in p['gpu_inventory'].splitlines()]
    selected=[row for row in inventory if row[0]==p['gpu_uuid']]
    require(len(selected)==1 and pci(selected[0][1])==pci(a['device']['pci']),'Selected GPU inventory PCI')
    rows=[dict(c,order='forward' if int(c['name'][:2])<4 else 'reverse') for c in a['cases'] if not c['name'].startswith('pilot')]
    require(all(x['wall_end']<=y['wall_start'] for x,y in zip(a['cases'],a['cases'][1:])),'Chronological cases')
    medians=[];ratios=[]
    for streams in (1,2):
        for n in (10000,100000):
            matches=[c for c in rows if c['streams']==streams and c['events']==n]
            require(len(matches)==2,'Two repeats')
            medians.append(dict(streams=streams,events=n,gib_s=statistics.median(c['gib_s'] for c in matches),hz=statistics.median(c['hz'] for c in matches),rates=[c['gib_s'] for c in matches]))
    for order in ('forward','reverse'):
        lookup={(c['streams'],c['events']):c for c in rows if c['order']==order}
        ratios.append(dict(order=order,stream2_over_stream1={str(n):lookup[2,n]['gib_s']/lookup[1,n]['gib_s'] for n in (10000,100000)},
            full_over_prefix={str(s):lookup[s,100000]['gib_s']/lookup[s,10000]['gib_s'] for s in (1,2)}))
    return dict(status='pass',job=a['job'],host=a['host'],mode=a['mode'],device=a['device'],selected_inventory=selected[0],allocated_mask=p['allocated_mask'],physical_cpus=a['physical_cpus'],
        audit_sha256=sha(root/'audit.json'),individuals=rows,medians=medians,paired_ratios=ratios,
        limits=['Local page cache cold on BD files only; SMD and Lustre server caches uncontrolled.',
                'Validation reads the full files before pilots; reverse 10k prefixes follow all full-file cases.',
                'Startup, allocations, fill, drain and barriers remain timed; short cases include a larger fixed-overhead fraction.',
                'Remote traffic unverified; no network-byte ratio or cache-hit fraction is inferred.',
                'Within-site comparison on Lustre; does not attribute the earlier SDF Weka discrepancy.',
                'Two repeats are descriptive measurements, not confidence intervals; descriptor staging only.'])
if __name__=='__main__':
    root=Path(sys.argv[1]);write(root/'summary.json',summarize(root))
