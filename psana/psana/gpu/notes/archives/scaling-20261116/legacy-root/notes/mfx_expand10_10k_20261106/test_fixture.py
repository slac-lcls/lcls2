"""Boundary and rejection checks for the sample10 wire rewrite."""
import struct
import unittest

import numpy as np
from psana import dgram
from psana.dgramedit import AlgDef, DetectorDef, DgramEdit

from fixture import (PIXELS, WORDS, config_transform, event_transform, metadata,
                     schedule, schemas)
from verify import expand, check_provenance
from fixture import HELPERS, SNAPSHOT, file_sha, children
from pathlib import Path
import copy


class Sample10Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = DgramEdit(transition_id=2, ts=1, bufsize=65536)
        det = cfg.Detector(DetectorDef('jungfrau', 'jungfrau', 'test-panel'), AlgDef('raw', 0, 2, 0),
                           {'raw': (np.uint16, 3), 'frame_cnt': (np.uint64, 0),
                            'timestamp': (np.uint64, 0), 'hotPixelThresh': (np.uint16, 0),
                            'numHotPixels': (np.uint32, 0), 'maxHotPixels': (np.uint32, 0)},
                           nodeId=5, namesId=10, segment=3)
        buf = bytearray(2*1024*1024)
        cfg.save(buf)
        cls.config = bytes(buf[:cfg.size])
        evt = DgramEdit(transition_id=12, ts=2, config_dgramedit=cfg, bufsize=len(buf))
        cls.raw = np.arange(PIXELS, dtype=np.uint16).reshape(1, 512, 1024)
        det.raw.raw = cls.raw
        for k, v in [('frame_cnt', 123456), ('timestamp', 987654), ('hotPixelThresh', 123),
                     ('numHotPixels', 2), ('maxHotPixels', 7)]:
            setattr(det.raw, k, v)
        evt.adddata(det.raw)
        evt.save(buf)
        cls.event = bytes(buf[:evt.size])
        cls.jf = schemas(cls.config)

    def test_full_words_and_tail(self):
        seed = self.raw.reshape(-1)[::10].copy()
        result = expand(seed).reshape(-1)
        np.testing.assert_array_equal(result, self.raw.reshape(-1)[np.arange(PIXELS)//10*10])
        self.assertEqual(len(seed), WORDS)
        np.testing.assert_array_equal(result[-8:], np.full(8, seed[-1], np.uint16))
        self.assertEqual(result.nbytes, PIXELS*2)

    def test_bad_seed_size_and_dtype(self):
        for seed in (np.zeros(WORDS-1, np.uint16), np.zeros(WORDS, np.uint8), np.zeros((1, WORDS), np.uint16)):
            with self.assertRaises(ValueError):
                expand(seed)

    def test_wire_roundtrip_and_scalars(self):
        _, meta = metadata(set(self.jf), 5)
        cfg = dgram.Dgram(view=config_transform(self.config, self.jf, meta))
        raw = event_transform(self.event, self.jf)
        obj = dgram.Dgram(config=cfg, view=raw).jungfrau[3].sim10
        np.testing.assert_array_equal(obj.seed, self.raw.reshape(-1)[::10])
        self.assertEqual(obj.frame_cnt, 123456)
        self.assertEqual(obj.timestamp, 987654)
        self.assertEqual(obj.maxHotPixels, 7)
        self.assertEqual(cfg.sim10meta[5].config.factor, 10)

    def test_missing_panel_rejected(self):
        with self.assertRaises(ValueError):
            event_transform(self.event, {**self.jf, 1291: self.jf[1290]})

    def test_broken_extent_rejected(self):
        raw = bytearray(self.event)
        struct.pack_into('<I', raw, 20, len(raw))
        with self.assertRaises(ValueError):
            event_transform(bytes(raw), self.jf)

    def test_shapes_after_data_rejected(self):
        # Reorder a valid panel's children without changing any extents.
        raw = bytearray(self.event)
        shape = next((p,n) for p,_,k,n in children(raw) if k==2)
        data = next((p,n) for p,_,k,n in children(raw) if k==3)
        p,n = shape; q,m = data
        self.assertEqual(p+n,q)
        swapped = bytes(raw[:p] + raw[q:q+m] + raw[p:p+n] + raw[q+m:])
        with self.assertRaisesRegex(ValueError, 'Shapes before Data'):
            event_transform(swapped, self.jf)

    def test_provenance_rejects_changed_writer_and_manifest(self):
        paths = [Path(__file__).with_name('fixture.py'), HELPERS/'xtc_fixture.py']
        good = {'code_sha256':{str(p):file_sha(p) for p in paths},
                'calibration_manifest_sha256':file_sha(SNAPSHOT/'manifest.json')}
        check_provenance(good)
        bad = copy.deepcopy(good);bad['code_sha256'][str(paths[0])]='0'*64
        with self.assertRaisesRegex(ValueError,'digest mismatch'):
            check_provenance(bad)
        bad = copy.deepcopy(good);bad['calibration_manifest_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'manifest digest mismatch'):
            check_provenance(bad)

    def test_schedule_rollover(self):
        rows = [{'service': s, 'timestamp': (10 << 32) + i*1000} for i, s in enumerate([2,4,6,8])]
        rows += [{'service': 12, 'timestamp': (10 << 32) + 999999900 - (9-i)*10000} for i in range(10)]
        rows += [{'service': 12, 'timestamp': (11 << 32)+10000}]
        rows += [{'service': s, 'timestamp': (11 << 32)+20000+i*1000} for i, s in enumerate([9,7,5])]
        result = schedule(rows)
        self.assertEqual([r['service'] for r in result[-3:]], [9,7,5])
        self.assertTrue(all(r['timestamp'] >> 32 == 11 for r in result[-3:]))


class LargeScheduleTests(unittest.TestCase):
    def test_10k_preserves_event_identity_and_closes(self):
        services = [2, 4, 6, 8] + [12]*10001 + [9, 7, 5]
        rows = [{'timestamp': (100 << 32) + i*10000, 'service': s} for i, s in enumerate(services)]
        planned = schedule(rows, 10000)
        self.assertEqual([r['timestamp'] for r in planned if r['service']==12],
                         [r['timestamp'] for r in rows if r['service']==12][:10000])
        self.assertEqual([r['service'] for r in planned[-3:]], [9, 7, 5])
        self.assertLess(planned[-1]['timestamp'], rows[10004]['timestamp'])
        self.assertEqual(len(planned), 10007)

    def test_oversized_and_unavailable_windows_rejected(self):
        for count in (0, 10001, 1.5):
            with self.assertRaises(ValueError):
                schedule([], count)
        with self.assertRaises(ValueError):
            schedule([], 10000)


if __name__ == '__main__':
    unittest.main()
