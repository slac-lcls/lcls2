import json,tempfile,unittest
from pathlib import Path
from data_contract import sha,identity
from requalify import verify_reuse


class ReuseVerifiedCopy(unittest.TestCase):
    def test_only_jf_identities_gate_reuse_and_receipt_is_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);files=[]
            for stream in range(10):
                for suffix in ('','.smd'):
                    p=root/('prjdatpsana-r0004-s%03d-c000%s.xtc2'%(stream,suffix))
                    p.write_bytes(b'original')
                    files.append(dict(destination=str(p),destination_identity=identity(p)))
            receipt=root/'staging.json';receipt.write_text(json.dumps(dict(status='pass',job='61939667',files=files)))
            plan=root/'plan.json';plan.write_text(json.dumps(dict(staging_receipt=str(receipt),pins={str(receipt):sha(receipt)})))
            verify_reuse(plan)
            Path(files[0]['destination']).write_bytes(b'non-JF content is excluded')
            verify_reuse(plan)
            jf=Path(files[10]['destination']);jf.write_bytes(b'JF changed')
            with self.assertRaisesRegex(ValueError,'Staged JF file changed'):verify_reuse(plan)
            receipt.write_text('{}')
            with self.assertRaisesRegex(ValueError,'Pinned recovery input changed'):verify_reuse(plan)
