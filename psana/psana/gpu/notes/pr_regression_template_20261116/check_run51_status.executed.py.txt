"""Read-only run-51 Configure identity and selected pixel-status audit."""
import ast, datetime, hashlib, json, os
from pathlib import Path
import numpy as np
import requests
from psana import dgram
from psana.pscalib.calib import MDBWebUtils as wu
from psana.pscalib.calib import CalibConstants as cc
root=Path(__file__).parent
base='https://pswww.slac.stanford.edu/calib_ws/'
segments={}
config_hashes={}
data=Path('/pscratch/sd/p/psdatmgr/psdm/mfx/mfx100848724/xtc/smalldata')
for path in sorted(data.glob('mfx100848724-r0051-*.smd.xtc2')):
 fd=os.open(path,os.O_RDONLY)
 try:
  cfg=dgram.Dgram(file_descriptor=fd)
  if hasattr(cfg.software,'jungfrau'):
   config_hashes[path.name]=hashlib.sha256(bytes(cfg)).hexdigest()
   for seg,value in cfg.software.jungfrau.items():
    assert seg not in segments
    segments[seg]=value.detid
 finally: os.close(fd)
assert sorted(segments)==list(range(32))
longname='jungfrau_'+'_'.join(segments[i] for i in sorted(segments))
cc.URL=base
_,dbname,col,query=wu.dbnames_collection_query(longname,exp='mfx100848724',ctype='pixel_status',run=51)
docs=wu.find_docs(dbname,col,query)
doc=wu.select_doc_in_run_range(docs,51)
assert doc and doc['longname']==longname and doc['data_type']=='ndarray'
(root/'selected-status-doc.json').write_text(json.dumps(doc,indent=2)+'\n')
print(json.dumps({'database':dbname,'collection':col,'doc_id':doc['_id'],'data_id':doc['id_data'],'run':doc['run'],'run_end':doc['run_end'],'dtype':doc['data_dtype'],'shape':doc['data_shape']}),flush=True)
shape=ast.literal_eval(doc['data_shape']); dtype=np.dtype(doc['data_dtype'])
assert shape==(3,32,512,1024) and dtype.kind in 'iu'
path=root/'run51-pixel-status.bin'
assert not path.exists()
h=hashlib.sha256(); size=0
with requests.get(base+dbname+'/gridfs/'+doc['id_data'],stream=True,timeout=(15,90)) as response:
 response.raise_for_status()
 with path.open('xb') as output:
  for block in response.iter_content(4<<20): output.write(block); h.update(block); size+=len(block)
assert size==int(np.prod(shape))*dtype.itemsize
status=np.memmap(path,dtype=dtype,mode='r',shape=shape)
counts=[]; effective=np.zeros(shape[1:],dtype=np.uint64)
for gain in range(3):
 effective |= status[gain]
 counts.append({'gain':gain,'total':int(status[gain].size),'zero':int(np.count_nonzero(status[gain]==0)),'nonzero':int(np.count_nonzero(status[gain]))})
values,count=np.unique(effective,return_counts=True)
report=dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),experiment='mfx100848724',run=51,detector='jungfrau',collection=col,database=dbname,selected_document={k:doc.get(k) for k in ('_id','id_data','run','run_end','version','data_dtype','data_shape')},payload_bytes=size,payload_sha256=h.hexdigest(),configure_sha256=config_hashes,gains=counts,effective_mask_policy='OR of all three gain planes; all nonzero status bits reject pixel',effective_pixels=int(effective.size),effective_good=int(np.count_nonzero(effective==0)),effective_bad=int(np.count_nonzero(effective)),effective_status_values={str(int(v)):int(n) for v,n in zip(values,count)},bad_per_panel=[int(np.count_nonzero(p)) for p in effective],scope='Selected status only; no calibration writes, image processing or benchmark. Detector identity and selection use installed PR299 psana runtime.')
(root/'run51-status-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
