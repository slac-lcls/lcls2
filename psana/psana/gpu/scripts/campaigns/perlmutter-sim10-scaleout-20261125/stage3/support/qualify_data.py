"""One-time compute-node transfer/SMD qualification. Never submit or transfer data."""
import argparse,hashlib,json,os,socket,struct,subprocess,sys,time
from pathlib import Path
from data_contract import validate_site,published_files,sha,identity,require
HERE=Path(__file__).resolve().parent


def records(path):
    with Path(path).open('rb',buffering=0) as stream:
        offset=0;size=os.fstat(stream.fileno()).st_size
        while offset<size:
            header=stream.read(24);require(len(header)==24,'Truncated header')
            timestamp,environment,source,damage,kind,extent=struct.unpack('<QIIHHI',header)
            length=extent+12
            require(kind==0 and damage==0 and 24<=length<=128<<20 and offset+length<=size,'Datagram extent/root')
            body=stream.read(length-24);require(len(body)==length-24,'Short datagram')
            yield offset,timestamp,(environment>>24)&15,header+body
            offset+=length
        require(offset==size,'Trailing data')


def scan_smd(path,expected):
    from psana import dgram
    digest=hashlib.sha256();config=None;events=[];transitions=[]
    for offset,timestamp,service,payload in records(path):
        digest.update(payload)
        if config is None:
            require(offset==0 and service==2,'Initial SMD Configure');config=dgram.Dgram(view=payload);continue
        require(service!=2,'One Configure per file')
        item=dgram.Dgram(view=payload,config=config)
        if service==12:
            entry=item.smdinfo[0].offsetAlg
            events.append((timestamp,int(entry.intOffset),int(entry.intDgramSize)))
        else:transitions.append((timestamp,service))
    require(digest.hexdigest()==expected,'Published SMD SHA256')
    require(len(events)==400000 and all(a[0]<b[0] for a,b in zip(events,events[1:])),'400k ordered SMD events')
    return events,transitions


def scan_payload(path,expected,smd,transitions,jf):
    from psana import dgram
    sys.path.insert(0,str(HERE/'base/vendor/sim10_compute_cube_20261115'))
    from reference import SeedReader
    digest=hashlib.sha256();config=None;events=0;control=[];panels=None;size=None
    for offset,timestamp,service,payload in records(path):
        digest.update(payload)
        if config is None:
            require(offset==0 and service==2,'Initial payload Configure');config=dgram.Dgram(view=payload);continue
        require(service!=2,'One Configure per file')
        if service==12:
            require(events<len(smd) and (timestamp,offset,len(payload))==smd[events],'SMD points to exact payload event')
            if jf:
                require(len(payload)>24,'Empty Jungfrau event')
                if size is None:size=len(payload)
                require(len(payload)==size,'Fixed JF event record size')
                if events in (0,399999):
                    layout=SeedReader._layout(payload,config)
                    selected=set(layout[-1])
                    if panels is None:panels=selected
                    require(selected==panels,'First/last panel schema')
            events+=1
        else:control.append((timestamp,service))
    require(events==400000 and control==transitions,'Exact events/transitions')
    require(digest.hexdigest()==expected,'Published payload SHA256')
    return dict(events=events,panels=sorted(panels) if jf else None,event_size=size)


def validate_jf_sizes(ranges):
    # Byte contract inherited from the qualified Stage 2 runner/audit.
    sizes=sorted(row['size'] for row in ranges.values())
    require(sizes==[524724,629664,734604,734604,734604] and sum(sizes)==3358200,
            'Published JF record sizes differ from frozen runner byte contract')


def qualify(site,output):
    require(os.environ.get('SLURM_JOB_ID') and socket.gethostname().startswith('nid'),'Heavy validation requires allocated compute node')
    require(not output.exists(),'Never overwrite qualification attempt')
    sums=validate_site(site);output.mkdir(parents=True)
    started=time.time();receipt=dict(status='running',site=site,job=os.environ['SLURM_JOB_ID'],host=socket.gethostname(),streams=list(range(5,10)),files=[])
    def save(): (output/'qualification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    save()
    try:
        root=Path(site['xtc_directory']);view=output/'view';(view/'smalldata').mkdir(parents=True)
        ranges={};selected=[];all_timestamps=None;all_panels=set()
        for stream in range(5,10):
            name='prjdatpsana-r0004-s%03d-c000.xtc2'%stream
            smdrel='smalldata/'+name.replace('.xtc2','.smd.xtc2')
            p=(root/name).resolve();s=(root/smdrel).resolve();before=identity(p);before_smd=identity(s)
            events,transitions=scan_smd(s,sums[smdrel]);stamps=[e[0] for e in events]
            if all_timestamps is None:all_timestamps=stamps
            require(stamps==all_timestamps,'All five Jungfrau streams aligned')
            result=scan_payload(p,sums[name],events,transitions,stream>=5)
            require(before==identity(p) and before_smd==identity(s),'Files changed during scan')
            receipt['files'].append(dict(path=str(p),sha256=sums[name],identity=before,
                                         smd=str(s),smd_sha256=sums[smdrel],**result));save()
            if stream>=5:
                require(not all_panels.intersection(result['panels']),'Duplicate physical panel')
                all_panels.update(result['panels'])
                (view/name).symlink_to(p);(view/'smalldata'/Path(smdrel).name).symlink_to(s)
                # Retain owner-scoped cache policy from Stage 2. Shared-owner
                # datasets need an explicitly reviewed policy, not an implicit copy.
                require(before['st_uid']==os.geteuid(),'Cold-cache input owner differs; resolve cache policy before preparing timings')
                layout=subprocess.check_output(['lfs','getstripe',str(p)],text=True)
                selected.append(dict(path=str(p),view=str(view/name),bytes=before['st_size'],identity=before,
                                     sha256=sums[name],smd=str(s),smd_sha256=sums[smdrel],layout_raw=layout))
                ranges[p.name]=dict(offsets=[e[1] for e in events],size=result['event_size'])
        require(all_panels==set(range(32)),'All 32 physical panels')
        validate_jf_sizes(ranges)
        manifest=json.loads((Path(site['calibration_snapshot'])/'manifest.json').read_text())
        for item in manifest['calibration']:
            if item['detector']=='jungfrau' and item['source_document']['ctype'] in ('pedestals','pixel_gain','pixel_offset'):
                require(sha(Path(site['calibration_snapshot'])/item['file'])==item['sha256'],'Frozen calibration payload hash')
        receipt.update(status='pass',finished=time.time(),seconds=time.time()-started,
                       sha256sums_sha256=sha(site['sha256sums']),qualified_events=400000)
        save()
        info=dict(exp='prjdatpsana',run=4,directory=str(view),timestamps=all_timestamps,
                  streams=list(range(5,10)),files=[r['path'] for r in selected],payload_records=selected,
                  media='lustre',qualified_events=400000,calibration_snapshot=site['calibration_snapshot'],
                  calibration_selection=site['calibration_selection'],calibration_selection_sha256=sha(site['calibration_selection']),
                  qualification=dict(status='pass',path=str(output/'qualification.json'),sha256=sha(output/'qualification.json')))
        (output/'inputs.json').write_text(json.dumps(info)+'\n');(output/'ranges.json').write_text(json.dumps(ranges)+'\n')
        return info
    except BaseException as error:receipt.update(status='failed',error=repr(error));save();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('site');parser.add_argument('output');a=parser.parse_args()
    qualify(json.loads(Path(a.site).read_text()),Path(a.output).resolve())
