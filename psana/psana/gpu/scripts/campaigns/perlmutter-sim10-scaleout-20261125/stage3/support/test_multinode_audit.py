"""Exercise composed independent audits with two hosts sharing PCI addresses."""
import ast,copy,hashlib,importlib.util,json,os,struct,sys,tempfile,unittest
from pathlib import Path
from compose import compose,HERE
from test_preparation import probes
from topology import build


class MultiNodeAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();root=Path(cls.temp.name)
        for n,b in compose().items():
            p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
        os.environ['SIM10_RUNTIME']='C';sys.path.insert(0,str(root))
        from audit import audit_case
        cls.audit=staticmethod(audit_case)
        from support import EVENT_BYTES,histogram
        tree=ast.parse((HERE/'base/test_audit_port.py').read_text())
        tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fixture']
        ns=dict(copy=copy,hashlib=hashlib,struct=struct,EVENT_BYTES=EVENT_BYTES,histogram=histogram)
        exec(compile(tree,'base fixture','exec'),ns);cls.original=staticmethod(ns['fixture'])

    @classmethod
    def tearDownClass(cls):sys.path.pop(0);cls.temp.cleanup()

    def fixture(self):
        report,case,info,ranges,_,_=self.original();template=report['rows'][2]
        hosts,p=probes(2);g=build(p,hosts);stamps=list(range(400))
        info['timestamps']=stamps
        for record in ranges.values():record['offsets']=[100+i*record['size'] for i in stamps]
        rows=[]
        for rank,host in enumerate(g['rank_hosts']):
            if rank<2:row=dict(rank=rank,host=host,affinity=g['mapping'][rank],timestamps=[],device=None,cube_verification=None)
            else:
                row=copy.deepcopy(template);ts=stamps[(rank-2)*50:(rank-1)*50]
                uuid=g['rank_uuids'][rank]
                row.update(rank=rank,host=host,affinity=g['mapping'][rank],timestamps=ts,
                           device=dict(mask=uuid,pci=g['pci_by_uuid'][uuid]),
                           granted_uuids=[g['rank_uuids'][r] for r,h in enumerate(g['rank_hosts']) if h==host and r>=2])
                row['callbacks'][0]['timestamps']=ts
                row['io']['range_sha256']={path:hashlib.sha256(b''.join(struct.pack('<QQQ',t,100+t*ranges[Path(path).name]['size'],ranges[Path(path).name]['size']) for t in ts)).hexdigest() for path in info['files']}
            row.update(node_observation=None,node_memory=[]);rows.append(row)
        for host in hosts:
            leader=next(r for r in rows if r['host']==host)
            leader['node_observation']=copy.deepcopy(report['observation'])
            for r in rows[2:]:
                if r['host']==host:
                    leader['node_memory'].append(dict(uuid=r['device']['mask'],errors=[],samples=[dict(time=t,total=40<<30,free=6<<30,used=34<<30) for t in (1.,2.)],max_used=34<<30,min_free=6<<30))
        report.update(events=400,active_gpus=8,hz=400.,payload_gib_s=8*report['payload_gib_s'],rows=rows,
                      node_events={h:200 for h in hosts},node_observations={r['host']:r['node_observation'] for r in rows if r['node_observation']})
        report['observation']=rows[0]['node_observation'];case.update(events=400,bds=8,gpus=8,nodes=2)
        return report,case,info,ranges,g['mapping'],g

    def test_good_two_node_receipt(self):self.assertEqual(self.audit(*self.fixture())['status'],'pass')

    def test_corruptions_rejected(self):
        for corruption in ('remote_cache','missing_remote','same_host_pci','memory_uuid','duplicate_work','range','rank_host','bytes'):
            with self.subTest(corruption=corruption):
                args=self.fixture();report=args[0];row=report['rows'][6]
                if corruption=='remote_cache':row['node_observation']['cache_gate']['attempts'][0]['scans'][1]['files'][0]['resident_pages']=1
                if corruption=='missing_remote':del report['node_observations'][row['host']]
                if corruption=='same_host_pci':report['rows'][7]['device']['pci']=row['device']['pci']
                if corruption=='memory_uuid':row['node_memory'][1]['uuid']=row['node_memory'][0]['uuid']
                if corruption=='duplicate_work':row['timestamps'][0]=0
                if corruption=='range':row['io']['range_sha256'][args[2]['files'][0]]='0'*64
                if corruption=='rank_host':row['host']=report['rows'][0]['host']
                if corruption=='bytes':row['io']['completed_bytes']-=1
                with self.assertRaises(ValueError):self.audit(*args)


if __name__=='__main__':unittest.main()
