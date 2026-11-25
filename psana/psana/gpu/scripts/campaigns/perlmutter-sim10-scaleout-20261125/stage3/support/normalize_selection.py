"""Normalize published run4 selection, retaining raw receipt provenance."""
import json
from pathlib import Path
from data_contract import require,sha,MANIFEST
HERE=Path(__file__).resolve().parent


def normalize(publication,snapshot):
    raw=publication/'calibration/verification.json';after=publication/'calibration/after.json'
    verification=json.loads(raw.read_text());documents=json.loads(after.read_text())
    require(verification['status']=='pass' and verification['database']=='cdb_prjdatpsana','Published verification')
    require(sha(snapshot/'manifest.json')==MANIFEST,'Snapshot manifest')
    expected={x['source_document']['ctype']:x['sha256'] for x in json.loads((snapshot/'manifest.json').read_text())['calibration'] if x['detector']=='jungfrau' and x['source_document']['ctype'] in ('pedestals','pixel_gain','pixel_offset')}
    selected=[x for x in verification['selected'] if x['detector']=='jungfrau' and x['run']==4 and x['ctype'] in expected]
    require(len(selected)==3 and {x['ctype'] for x in selected}==set(expected),'Exactly three run4 selections')
    records={}
    for item in selected:
        matches=[d for rows in documents.values() for d in rows if d['_id']==item['id_doc']]
        require(len(matches)==1,'Unique published calibration document');d=matches[0];meta=json.loads(d['extpars'])
        require(d['experiment']=='prjdatpsana' and d['detname']=='jungfrau' and d['ctype']==item['ctype'],'Published selection identity')
        require(d['id_data']==item['id_data'] and d['run']<=4<=d['run_end'],'Published validity and payload ID')
        require(meta['snapshot_manifest_sha256']==MANIFEST and meta['payload_sha256']==item['sha256']==expected[item['ctype']],'Frozen calibration payload match')
        records[item['ctype']]=dict(document_id=d['_id'],id_data=d['id_data'],run=d['run'],run_end=d['run_end'],
                                  payload_sha256=item['sha256'],source_receipt=str(raw),source_receipt_sha256=sha(raw),
                                  document_receipt=str(after),document_receipt_sha256=sha(after))
    return dict(status='pass',experiment='prjdatpsana',run=4,database='cdb_prjdatpsana',manifest_sha256=MANIFEST,records=records)


if __name__=='__main__':
    snapshot=Path('/pscratch/sd/m/monarin/psana2-gpu/sim10-baseline-transfer-20261115/package/calibration')
    result=normalize(HERE/'evidence/publication',snapshot)
    (HERE/'evidence/calibration-selection.json').write_text(json.dumps(result,indent=2)+'\n')
