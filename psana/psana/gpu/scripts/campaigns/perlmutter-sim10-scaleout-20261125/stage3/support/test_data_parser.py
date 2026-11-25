import struct,tempfile,unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
from qualify_data import records,validate_jf_sizes,scan_payload


class DatagramHeaders(unittest.TestCase):
    def test_empty_jf_event_rejected_even_without_damage_flag(self):
        config=struct.pack('<QIIHHI',123,2<<24,0,0,0,12)
        empty=struct.pack('<QIIHHI',456,12<<24,0,0,0,12)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'jf.xtc2';p.write_bytes(config+empty)
            with patch.dict('sys.modules', {'psana':SimpleNamespace(dgram=SimpleNamespace(Dgram=lambda **kw:object())),
                                            'reference':SimpleNamespace(SeedReader=object)}):
                with self.assertRaisesRegex(ValueError,'Empty Jungfrau event'):
                    scan_payload(p,'unused',[(456,24,24)],[],True)

    def test_valid_records_keep_offsets_and_timestamps(self):
        a=struct.pack('<QIIHHI',123,2<<24,0,0,0,12)
        b=struct.pack('<QIIHHI',456,12<<24,0,0,0,16)+b'abcd'
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'test.xtc2';p.write_bytes(a+b)
            result=list(records(p));self.assertEqual([(r[0],r[1],r[2],len(r[3])) for r in result],[(0,123,2,24),(24,456,12,28)])

    def test_published_sizes_must_match_frozen_byte_contract(self):
        sizes=[524724,629664,734604,734604,734604]
        validate_jf_sizes({i:dict(size=n) for i,n in enumerate(sizes)})
        # Equal total bytes cannot hide different physical read sizes.
        sizes[0]+=4;sizes[1]-=4
        with self.assertRaises(ValueError):
            validate_jf_sizes({i:dict(size=n) for i,n in enumerate(sizes)})

    def test_truncation_damage_and_bad_extents_rejected(self):
        for content in (b'x'*20,struct.pack('<QIIHHI',1,12<<24,0,1,0,12),
                        struct.pack('<QIIHHI',1,12<<24,0,0,0,8),
                        struct.pack('<QIIHHI',1,12<<24,0,0,0,128<<20)):
            with self.subTest(content=content),tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'test.xtc2';p.write_bytes(content)
                with self.assertRaises(ValueError):list(records(p))
