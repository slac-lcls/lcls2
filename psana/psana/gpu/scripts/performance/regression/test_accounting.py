import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import native_contract as nc
from accounting_probe import check_control


def snapshot(value=0,timestamp=1):
    return dict(monotonic=timestamp,unavailable={},interfaces={'hsn0':dict(device='/pci/0',samples={
        d:dict(value=value,timestamp_ns=timestamp) for d in ('rx','tx')})})


class TrafficTests(unittest.TestCase):
    def test_parser_and_failed_sampling(self):
        self.assertEqual(nc.parse_cxi_counter('12@123.000000456\n'),dict(value=12,timestamp_ns=123000000456))
        for bad in ('-1@123.000000000','12','12@123.4','1@nan.000000000','12@123.000000000 junk'):
            with self.subTest(bad=bad),self.assertRaises(ValueError):nc.parse_cxi_counter(bad)
        with patch.object(Path,'read_text',side_effect=PermissionError('denied')):
            observed=nc.cxi_snapshot(['hsn0'])
            self.assertIn('hsn0',observed['unavailable'])
            with self.assertRaises(ValueError):nc.cxi_delta(observed,observed)
        with self.assertRaises(ValueError):nc.cxi_snapshot(['../bad'])
        with patch.object(Path,'resolve',return_value=Path('/same')),patch.object(Path,'read_text',return_value='1@123.000000456'):
            with self.assertRaisesRegex(ValueError,'duplicate'):nc.cxi_snapshot(['hsn0','hsn1'])

    def test_delta_rejects_reset_stale_missing_or_changed_device(self):
        before,after=snapshot(10,1),snapshot(100,2)
        self.assertEqual(nc.cxi_delta(before,after),dict(rx_bytes=90,tx_bytes=90))
        for kind in ('reset','timestamp','device','missing','unavailable','clock'):
            bad=copy.deepcopy(after)
            if kind=='reset':bad['interfaces']['hsn0']['samples']['rx']['value']=9
            if kind=='timestamp':bad['interfaces']['hsn0']['samples']['rx']['timestamp_ns']=1
            if kind=='device':bad['interfaces']['hsn0']['device']='/pci/1'
            if kind=='missing':bad['interfaces']={}
            if kind=='unavailable':bad['unavailable']={'hsn1':'missing'}
            if kind=='clock':bad['monotonic']=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):nc.cxi_delta(before,bad)

    def test_control_rejects_false_coverage_and_changed_data(self):
        nic=dict(unavailable={},interfaces={'hsn0':dict(rx=0,tx=0)})
        rows=[]
        for kind in ('idle','cold','warm','cold','warm'):
            cache=dict(measured_bytes=100,resident_bytes_before=100 if kind=='warm' else 0)
            rows.append(dict(kind=kind,cxi_before=snapshot(0,1),cxi_after=snapshot(110 if kind=='cold' else 0,2),
                nic_before=nic,nic_after=nic,cache=dict(observations=[cache,cache]) if kind=='cold' else cache,bytes=100,sha256='same'))
        check_control(copy.deepcopy(rows),100)
        for kind in ('cold_missing','cold_excess','warm_traffic','idle','warm_cache','cold_cache','bytes','hash','sequence'):
            bad=copy.deepcopy(rows)
            if kind=='cold_missing':bad[1]['cxi_after']=snapshot(1,2)
            if kind=='cold_excess':bad[1]['cxi_after']=snapshot(201,2)
            if kind=='warm_traffic':bad[2]['cxi_after']=snapshot(11,2)
            if kind=='idle':bad[0]['cxi_after']=snapshot(11,2)
            if kind=='warm_cache':bad[2]['cache']['resident_bytes_before']=99
            if kind=='cold_cache':bad[1]['cache']['observations'][0]['resident_bytes_before']=1
            if kind=='bytes':bad[1]['bytes']=99
            if kind=='hash':bad[2]['sha256']='different'
            if kind=='sequence':bad=bad[:-1]
            with self.subTest(kind=kind),self.assertRaises(ValueError):check_control(bad,100)



class FullPrefixTests(unittest.TestCase):
    def test_full_prefix_compares_accepted_hash_and_verifies_all_headers(self):
        import hashlib
        import struct
        import run_smoke as smoke
        from prepare_inputs import stamp
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); data=root/'private';data.mkdir(mode=0o700);(data/'smalldata').mkdir()
            records=[[i+1,0,i*24,24,12] for i in range(10000)]
            payload=b''.join(struct.pack('<6I',i+1,0,12<<24,0,0,12) for i in range(10000))
            ref=dict(schema='run51-raw-reference-v1',events=[[i+1,0,120] for i in range(10000)])
            plan=dict(schema='run51-input-plan-v1',reference=ref,streams=[])
            manifest=dict(schema='run51-private-input-v2',private_copy=True,root=str(data),files=[])
            for stream in (3,5,7,8,9):
                entries={}
                for kind in ('bigdata','smd'):
                    name='mfx100848724-r0051-s%03d-c000%s.xtc2'%(stream,'.smd' if kind=='smd' else '')
                    path=data/('smalldata' if kind=='smd' else '')/name
                    path.write_bytes(b'smd' if kind=='smd' else payload);path.chmod(0o444)
                    entries[kind]=dict(path=str(path),sha256=smoke.file_hash(path),**stamp(path))
                entries['bigdata'].update(ranges=[[0,len(payload)]],range_bytes=len(payload))
                manifest['files'].append(dict(stream=stream,**entries))
                plan['streams'].append(dict(stream=stream,smd=entries['smd'],events=records))
            nc.write(root/'reference.json',ref);nc.write(root/'plan.json',plan)
            manifest['event_reference_sha256']=smoke.file_hash(root/'reference.json')
            nc.write(root/'manifest.json',manifest)
            spec={k:smoke.artifact(root/(k+'.json')) for k in ('reference','plan','manifest')}
            receipt=dict(complete=True,events=10000,manifest_sha256=spec['manifest']['sha256'],
                         plan_sha256=spec['plan']['sha256'],event_reference_sha256=spec['reference']['sha256'])
            nc.write(root/'preparation.json',receipt);spec['preparation']=smoke.artifact(root/'preparation.json')
            nc.write(root/'spec.json',spec)
            smoke.prefix(str(root/'spec.json'),smoke.file_hash(root/'spec.json'),root/'out',10000)
            done=nc.read(root/'out/complete.json')
            self.assertEqual(done['headers_verified'],50000)
            self.assertEqual(done['requested_bytes'],1200000)
            smoke.verify_prefix(root/'out',spec,10000)
            with self.assertRaises(ValueError):smoke.verify_prefix(root/'out',spec,1000)
            manifest['files'][0]['bigdata']['sha256']='0'*64
            (root/'manifest.json').unlink();nc.write(root/'manifest.json',manifest);spec['manifest']=smoke.artifact(root/'manifest.json')
            receipt['manifest_sha256']=spec['manifest']['sha256'];(root/'preparation.json').unlink();nc.write(root/'preparation.json',receipt)
            spec['preparation']=smoke.artifact(root/'preparation.json');(root/'spec.json').unlink();nc.write(root/'spec.json',spec)
            with self.assertRaisesRegex(ValueError,'full-prefix hash'):
                smoke.prefix(str(root/'spec.json'),smoke.file_hash(root/'spec.json'),root/'bad',10000)
            self.assertFalse((root/'bad/complete.json').exists())

if __name__=='__main__':unittest.main()
