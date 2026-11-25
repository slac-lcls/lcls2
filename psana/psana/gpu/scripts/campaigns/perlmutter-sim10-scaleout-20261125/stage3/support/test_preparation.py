"""CPU tests for host-scoped invariants and fail-closed preparation."""
import copy,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from design import plans,cases,launch
from topology import build,rank_hosts,audit_placement
from data_contract import published_files,checksums,validate_site
from analyze import summarize,windows
from compose import compose


def probes(nodes):
    hosts=['nid%06d'%i for i in range(nodes)];result=[]
    for rank,host in enumerate(rank_hosts(hosts)):
        # PCI bus numbers and CPU IDs intentionally repeat across hosts.
        result.append(dict(rank=rank,host=host,cpus=list(range(128)),
            topology={str(c):['0',str(c%64)] for c in range(128)},
            gpus=[dict(uuid='GPU-'+host+'-'+str(g),pci='0000:%02x:00.0'%g) for g in range(4)]))
    return hosts,result


class Preparation(unittest.TestCase):
    def test_matrix_and_budget(self):
        ps=plans();self.assertEqual([p['ranks'] for p in ps],[6,10,18])
        self.assertEqual(sum(c['kind']=='full' for p in ps for c in p['cases']),36)
        for p in ps:
            self.assertEqual(p['walltime'],'03:16:00')
            self.assertLess(sum(p['budget'].values()),14400)
            self.assertIn('--qos=interactive',launch(p))
            self.assertFalse(p['submission_authorized'])
            self.assertTrue(all(c['events']==100000*p['nodes'] for c in p['cases'] if c['kind']!='acceptance'))

    def test_repeated_bus_ids_are_host_local(self):
        for nodes in (1,2,4):
            hosts,p=probes(nodes);placement=build(p,hosts)
            self.assertEqual(len(placement['rank_uuids']),4*nodes+2)
            rows=[dict(rank=r,host=h,affinity=placement['mapping'][r],timestamps=[] if r<2 else [r],
                       device=None if r<2 else dict(pci=placement['pci_by_uuid'][placement['rank_uuids'][r]]))
                  for r,h in enumerate(placement['rank_hosts'])]
            self.assertEqual(audit_placement(rows,placement),{h:4 for h in hosts})

    def test_wrong_node_layout_rejected(self):
        hosts,p=probes(2);p[6]['host']=hosts[0]
        with self.assertRaises(ValueError):build(p,hosts)

    def test_duplicate_gpu_rejected(self):
        hosts,p=probes(2)
        for row in p[6:]:row['gpus'][0]['uuid']=p[0]['gpus'][0]['uuid']
        with self.assertRaises(ValueError):build(p,hosts)

    def test_physical_cores_not_smt_count(self):
        hosts,p=probes(1)
        for row in p:row['topology']={str(c):['0',str(c%16)] for c in range(128)}
        with self.assertRaises(ValueError):build(p,hosts)

    def test_masks_must_have_common_cores(self):
        hosts,p=probes(1);p[0]['cpus']=list(range(10))
        with self.assertRaises(ValueError):build(p,hosts)

    def test_unexercised_bd_rejected(self):
        hosts,p=probes(1);placement=build(p,hosts)
        rows=[dict(rank=r,host=h,affinity=placement['mapping'][r],timestamps=[],device=None if r<2 else dict(pci=str(r))) for r,h in enumerate(rank_hosts(hosts))]
        with self.assertRaises(ValueError):audit_placement(rows,placement)

    def test_incomplete_or_unsafe_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'SHA256SUMS'
            path.write_text('0'*64+'  ../outside\n')
            with self.assertRaises(ValueError):checksums(path)
            path.write_text('0'*64+'  '+published_files()[0]+'\n')
            with self.assertRaises(ValueError):checksums(path)
            path.write_text('\n'.join('0'*64+'  '+n for n in published_files()))
            self.assertEqual(len(checksums(path)),20)

    def test_no_historical_substitution(self):
        with self.assertRaises(ValueError):validate_site(dict(exp='mfx101555026',run=10))
        with self.assertRaises(ValueError):validate_site(dict(exp='prjdatpsana',run=4))

    def test_analysis_and_anchor_rejection(self):
        state=dict(job='example',hosts=['nid000000'],completed=[dict(case=c,seconds=60.) for c in cases(1)])
        result=summarize(state);self.assertEqual(result['results'][0]['median_rate'],100000/60)
        state['completed'][-1]['seconds']=70
        with self.assertRaises(ValueError):summarize(state)

    def test_short_diagnostic_inconclusive(self):
        self.assertEqual(windows([],40)['status'],'inconclusive')
        self.assertEqual(len(windows([],60)['rates']),5)

    def test_composed_syntax_and_critical_guards(self):
        f=compose()
        for name,data in f.items():
            if name.endswith('.py'):compile(data,name,'exec')
        self.assertIn(b'if node_leader:',f['runner.py'])
        self.assertNotIn(b"len({entry['device']['pci']",f['runner.py'])
        self.assertIn(b'Every node cache observation',f['audit.py'])
        self.assertIn(b'Four distinct assigned memory watches',f['audit.py'])


if __name__=='__main__':unittest.main()
