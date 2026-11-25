"""Fail-closed published run-4 contract; no historical-data substitution."""
import hashlib,json,os,subprocess
from pathlib import Path
MANIFEST='90a27eed3679111aa24574c09b5505be6b8775429e5b8de10dc0c9de117b983d'


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8<<20),b''):h.update(part)
    return h.hexdigest()


def identity(path):
    s=Path(path).stat()
    return {k:getattr(s,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_uid')}


def published_files():
    return [prefix+'prjdatpsana-r0004-s%03d-c000%s.xtc2'%(stream,suffix)
            for stream in range(10) for prefix,suffix in [('', ''),('smalldata/', '.smd')]]


def checksums(path):
    result={}
    for line in Path(path).read_text().splitlines():
        if not line.strip():continue
        digest,name=line.split(maxsplit=1);name=name.lstrip('*')
        if name.startswith('./'):name=name[2:]
        require(len(digest)==64 and all(c in '0123456789abcdef' for c in digest),'SHA256 format')
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Relative published path')
        require(name not in result,'Duplicate checksum name');result[name]=digest
    require(set(published_files())<=set(result),'All twenty published run-4 hashes required')
    return result


def validate_site(site):
    require(site.get('exp')=='prjdatpsana' and site.get('run')==4,'Published run4 only')
    for key in ('xtc_directory','sha256sums','calibration_snapshot','calibration_selection'):
        require(isinstance(site.get(key),str) and Path(site[key]).is_absolute(),'Missing site path: '+key)
        require(Path(site[key]).exists(),'Unavailable site path: '+key)
    root=Path(site['xtc_directory']);sums=checksums(site['sha256sums'])
    for rel in published_files():
        require((root/rel).is_file() and os.access(root/rel,os.R_OK),'Unreadable published file: '+rel)
        if rel.startswith('prjdatpsana-r0004-s00') and int(rel.split('-s')[1][:3])>=5:
            require((root/rel).stat().st_uid==os.geteuid(),'Shared-owner data: resolve owner-scoped cache policy before freezing attempts')
    snapshot=Path(site['calibration_snapshot'])
    require(sha(snapshot/'manifest.json')==MANIFEST,'Frozen calibration manifest')
    # A reviewed metadata-only receipt must tie published experiment selection to
    # the frozen payloads. The snapshot alone proves no run-4 DB selection.
    receipt=json.loads(Path(site['calibration_selection']).read_text())
    require(receipt['status']=='pass' and receipt['experiment']=='prjdatpsana' and receipt['run']==4,'Published calibration selection')
    require(receipt['manifest_sha256']==MANIFEST and receipt['database']=='cdb_prjdatpsana','Calibration selection provenance')
    expected={x['source_document']['ctype']:x for x in json.loads((snapshot/'manifest.json').read_text())['calibration'] if x['detector']=='jungfrau' and x['source_document']['ctype'] in ('pedestals','pixel_gain','pixel_offset')}
    require(set(expected)=={'pedestals','pixel_gain','pixel_offset'},'Three calibration payloads')
    require(set(receipt['records'])==set(expected),'Three selected calibration records')
    for key,item in expected.items():
        selected=receipt['records'][key]
        require(selected['payload_sha256']==item['sha256'] and bool(selected['id_data']),'Selected calibration payload identity')
        require(int(selected['run'])<=4<=int(selected['run_end']),'Published run4 validity')
        require(selected.get('document_id') and selected.get('source_receipt'),'Selection document/evidence link')
        require(sha(selected['source_receipt'])==selected['source_receipt_sha256'],'Selection source receipt hash')
    return sums


def input_info():
    path=Path(os.environ['SIM10_INPUTS'])
    return json.loads(path.read_text())


def verify_inputs(info):
    require(info['exp']=='prjdatpsana' and info['run']==4 and info['qualified_events']==400000,'Qualified run4 identity')
    require(info['qualification']['status']=='pass' and len(info['timestamps'])==400000,'Full data qualification')
    require(len(info['files'])==len(info['payload_records'])==5,'JF-only selection')
    require(info['streams']==list(range(5,10)),'Selected stream set')
    require(sha(Path(info['calibration_snapshot'])/'manifest.json')==MANIFEST,'Calibration manifest changed')
    require(sha(info['calibration_selection'])==info['calibration_selection_sha256'],'Calibration selection changed')
    # Complete transfer was hashed on compute before this immutable manifest.
    # Recheck metadata/layout and SMD content without rehashing TBs per case.
    for r in info['payload_records']:
        require(Path(r['view']).resolve()==Path(r['path']),'View changed')
        require(identity(r['path'])==r['identity'],'Payload identity changed')
        require(sha(r['smd'])==r['smd_sha256'],'SMD changed')
        require(subprocess.check_output(['lfs','getstripe',r['path']],text=True)==r['layout_raw'],'Lustre layout changed')
