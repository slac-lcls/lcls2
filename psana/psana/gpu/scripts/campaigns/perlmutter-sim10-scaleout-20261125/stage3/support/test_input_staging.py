import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from stage_input import copy_one
from freeze_input import verify_publication
from normalize_selection import normalize,HERE


class InputStaging(unittest.TestCase):
    def test_copy_matches_publication_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);s=root/'source';d=root/'dest';s.write_bytes(b'fixture payload')
            expected=dict(bytes=s.stat().st_size,sha256=hashlib.sha256(s.read_bytes()).hexdigest())
            row=copy_one(s,d,expected);self.assertEqual(d.read_bytes(),s.read_bytes());self.assertEqual(row['sha256'],expected['sha256'])
            with self.assertRaises(FileExistsError):copy_one(s,d,expected)
            self.assertEqual(d.read_bytes(),s.read_bytes())

    def test_bad_size_and_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);s=root/'source';s.write_bytes(b'fixture')
            with self.assertRaises(ValueError):copy_one(s,root/'bad-size',dict(bytes=8,sha256='0'*64))
            self.assertFalse((root/'bad-size').exists())
            with self.assertRaises(ValueError):copy_one(s,root/'bad-hash',dict(bytes=7,sha256='0'*64))

    def test_publication_receipt_rejects_changed_evidence(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'publication';shutil.copytree(HERE/'evidence/publication',root)
            verify_publication(root)
            f=root/'SHA256SUMS';f.write_text(f.read_text()+'\n')
            with self.assertRaises(ValueError):verify_publication(root)

    def test_cancelled_copy_is_not_accepted(self):
        import threading
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.write_bytes(b'data')
            stop=threading.Event();stop.set()
            with self.assertRaises(InterruptedError):copy_one(source,root/'dest',dict(bytes=4,sha256=hashlib.sha256(b'data').hexdigest()),stop)

    def test_run4_selection_and_rejection(self):
        import shutil
        snapshot=Path('/pscratch/sd/m/monarin/psana2-gpu/sim10-baseline-transfer-20261115/package/calibration')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'publication';shutil.copytree(HERE/'evidence/publication',root)
            result=normalize(root,snapshot);self.assertEqual(set(result['records']),{'pedestals','pixel_gain','pixel_offset'})
            f=root/'calibration/verification.json';original=json.loads(f.read_text())
            for mode in ('missing_run4','wrong_hash'):
                d=copy.deepcopy(original)
                if mode=='missing_run4':d['selected']=[x for x in d['selected'] if x['run']!=4]
                else:
                    for x in d['selected']:
                        if x['run']==4 and x['detector']=='jungfrau':x['sha256']='0'*64
                f.write_text(json.dumps(d))
                with self.assertRaises(ValueError):normalize(root,snapshot)


if __name__=='__main__':unittest.main()
