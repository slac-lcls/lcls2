"""Focused CPU tests; run in the native psana environment with TMPDIR on scratch."""
import hashlib
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from reference import CpuCalibration, PIXELS, SHAPE, WORDS, SeedReader, samples


def xtc(kind, payload=b'', source=0):
    return struct.pack('<IHHI', source, 0, kind, 12 + len(payload)) + payload


def record(timestamp, payload=b'', service=12, environment=0):
    return struct.pack('<QI', timestamp, service << 24 | environment) + xtc(0, payload)


def configuration(panels):
    from psana.dgramedit import AlgDef, DetectorDef, DgramEdit
    editor = DgramEdit(transition_id=2, ts=1, bufsize=1 << 20)
    for panel in panels:
        editor.Detector(DetectorDef('jungfrau', 'jungfrau', 'test'), AlgDef('sim10', 1, 0, 0),
                        {'prefix': (np.uint32, 0), 'seed': (np.uint16, 1), 'suffix': (np.uint64, 0)},
                        nodeId=0, namesId=panel, segment=panel)
    buffer = bytearray(1 << 20)
    editor.save(buffer)
    return bytes(buffer[:editor.size])


def event(timestamp, panels, nested=False):
    children = []
    for panel in panels:
        values = ((np.arange(WORDS, dtype=np.uint32) * 127 + panel * 911 + timestamp) % 65536).astype('<u2')
        data = struct.pack('<I', timestamp) + values.tobytes() + struct.pack('<Q', timestamp + panel)
        children.append(xtc(1, xtc(2, struct.pack('<5I', WORDS, 0, 0, 0, 0)) + xtc(3, data), panel))
    payload = b''.join(children)
    return record(timestamp, xtc(0, payload) if nested else payload, environment=(timestamp & 255) << 16)


class SeedReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel_groups = [list(range(15, -1, -1)), list(range(31, 15, -1))]
        cls.configurations = [configuration(panels) for panels in cls.panel_groups]
        cls.events = [[event(timestamp, panels, stream_index == 1) for timestamp in (100, 200, 300)]
                      for stream_index, panels in enumerate(cls.panel_groups)]

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='seed-reference-')
        self.addCleanup(self.directory.cleanup)
        self.paths = [Path(self.directory.name) / f'stream-{index}.xtc2' for index in range(2)]
        self.write_streams()

    def write_streams(self, replacements=None):
        for stream_index, path in enumerate(self.paths):
            events = self.events[stream_index] if replacements is None else replacements[stream_index]
            path.write_bytes(self.configurations[stream_index] + record(2, service=4) +
                             events[0] + record(150, xtc(0), service=10) + b''.join(events[1:]))

    def reader(self):
        reader = SeedReader(self.paths, (100, 200, 300))
        self.addCleanup(reader.close)
        return reader

    def test_native_sampled_full_bitwise_and_read_only(self):
        before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in self.paths]
        reader = self.reader()
        selected = np.concatenate((samples()[::-1], [9, 0, 10, 9, PIXELS - 1]))
        for timestamp in (300, 100, 200, 100):
            actual = reader.raw(timestamp, selected)
            expected = reader.raw(timestamp).reshape(32, PIXELS)[:, selected]
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual.dtype, np.uint16)
        self.assertEqual(reader.raw(100, np.array([], dtype=np.int64)).shape, (32, 0))
        after = [hashlib.sha256(path.read_bytes()).hexdigest() for path in self.paths]
        self.assertEqual(before, after)
        self.assertNotEqual(reader.layouts[0][2][15], reader.layouts[1][2][31])

    def test_warm_reads_are_controls_and_unique_word_runs_only(self):
        reader = self.reader()
        selected = samples()
        reader.raw(100, selected)
        native_read = os.pread
        with patch('reference.os.pread', wraps=native_read) as reads:
            reader.raw(300, selected)
        unique_words = len(np.unique(selected // 10))
        expected = sum(24 + sum(len(content) for _, content in controls) + len(seeds) * unique_words * 2
                       for _, controls, seeds in reader.layouts)
        self.assertEqual(sum(call.args[1] for call in reads.call_args_list), expected)
        self.assertLess(expected, 8192)
        self.assertLess(max(call.args[1] for call in reads.call_args_list), 256)
        self.assertGreater(reader.offsets[0][200] - reader.offsets[0][100], len(self.events[0][0]))

    def test_changed_same_size_layout_rejected_but_full_path_works(self):
        events = [list(values) for values in self.events]
        events[0][1] = event(200, self.panel_groups[0][::-1])
        self.assertEqual(len(events[0][1]), len(self.events[0][1]))
        self.write_streams(events)
        reader = self.reader()
        reader.raw(100, samples())
        with self.assertRaisesRegex(ValueError, 'layout changed'):
            reader.raw(200, samples())
        actual = reader.raw(200)
        expected = ((np.arange(WORDS, dtype=np.uint32) * 127 + 15 * 911 + 200) % 65536).astype(np.uint16)
        np.testing.assert_array_equal(actual[15].reshape(-1), np.repeat(expected, 10)[:PIXELS])

    def test_changed_structural_bytes_rejected_on_every_record(self):
        changes = [(24, '<I', 14), (28, '<H', 1), (30, '<H', 0), (32, '<I', 12),
                   (44, '<I', 31), (48, '<I', WORDS - 1), (76, '<I', 12)]
        for offset, encoding, value in changes:
            with self.subTest(offset=offset):
                events = [list(values) for values in self.events]
                changed = bytearray(events[0][1])
                struct.pack_into(encoding, changed, offset, value)
                events[0][1] = bytes(changed)
                self.write_streams(events)
                reader = self.reader()
                reader.raw(100, samples())
                with self.assertRaisesRegex(ValueError, 'layout changed'):
                    reader.raw(200, samples())
                reader.close()

    def test_malformed_witness_rejected_before_native_event_decode(self):
        reader = self.reader()
        changes = [(24, '<I', 255), (28, '<H', 1), (30, '<H', 4), (32, '<I', 11),
                   (32, '<I', len(self.events[0][0])), (48, '<I', WORDS + 1), (76, '<I', 12)]
        for offset, encoding, value in changes:
            with self.subTest(offset=offset, value=value):
                payload = bytearray(self.events[0][0])
                struct.pack_into(encoding, payload, offset, value)
                with patch('psana.dgram.Dgram', side_effect=AssertionError('Native decode must not run')):
                    with self.assertRaises(ValueError):
                        SeedReader._layout(bytes(payload), reader.configs[0])

    def test_native_view_must_share_buffer_and_stay_in_its_data(self):
        from psana import dgram
        reader = self.reader()
        payload = self.events[0][0]
        for copied in (True, False):
            with self.subTest(copied=copied):
                decoded = dgram.Dgram(config=reader.configs[0], view=payload)
                original = decoded.jungfrau[15].sim10.seed
                decoded.jungfrau[15].sim10.seed = (original.copy() if copied else
                                                 np.frombuffer(payload, dtype='<u2', count=WORDS))
                with patch('psana.dgram.Dgram', return_value=decoded):
                    with self.assertRaisesRegex(ValueError, 'native seed'):
                        SeedReader._layout(payload, reader.configs[0])

    def test_changed_configure_and_initial_service_rejected(self):
        self.paths[0].write_bytes(self.configurations[0] + self.events[0][0] + self.configurations[0] +
                                 b''.join(self.events[0][1:]))
        with self.assertRaisesRegex(ValueError, 'Configure changed'):
            self.reader()
        self.paths[0].write_bytes(record(1, service=4))
        with self.assertRaisesRegex(ValueError, 'start with Configure'):
            self.reader()

    def test_source_changed_after_index_rejected(self):
        reader = self.reader()
        with self.paths[0].open('r+b') as stream:
            stream.seek(8)
            stream.write(b'\x01')
        with self.assertRaisesRegex(ValueError, 'Source file changed'):
            reader.raw(100, samples())

    def test_truncated_index_and_record_reads(self):
        for suffix in (b'', record(100)[:23], self.events[0][0][:-1]):
            with self.subTest(size=len(suffix)):
                self.paths[0].write_bytes(self.configurations[0] + suffix)
                with self.assertRaisesRegex(ValueError, 'Short source|Truncated source'):
                    SeedReader(self.paths, (100,))
        self.paths[0].write_bytes(record(100)[:-1])
        with self.paths[0].open('rb') as stream:
            with self.assertRaisesRegex(ValueError, 'Short source'):
                SeedReader._record(stream, 0)
        self.paths[0].write_bytes(self.events[0][0][:-1])
        with self.paths[0].open('rb') as stream:
            with self.assertRaisesRegex(ValueError, 'Truncated source'):
                SeedReader._record(stream, 0)

    def test_short_sample_word_or_control_read_rejected(self):
        reader = self.reader()
        reader.raw(100, samples())
        native_read = os.pread
        for target_size in (2, 24):
            def short_read(descriptor, size, offset):
                payload = native_read(descriptor, size, offset)
                return payload[:-1] if size == target_size else payload
            with self.subTest(target_size=target_size), patch('reference.os.pread', side_effect=short_read):
                with self.assertRaisesRegex(ValueError, 'Short source read'):
                    reader.raw(200, samples())

    def test_timestamp_and_service_checked_at_sample_time(self):
        reader = self.reader()
        reader.raw(100, samples())
        native_read = os.pread
        for changed in (struct.pack('<Q', 301), None):
            def wrong_header(descriptor, size, offset):
                payload = bytearray(native_read(descriptor, size, offset))
                if size == 24:
                    if changed is None:
                        payload[11] = 10
                    else:
                        payload[:8] = changed
                return bytes(payload)
            with self.subTest(changed=changed), patch('reference.os.pread', side_effect=wrong_header):
                with self.assertRaisesRegex(ValueError, 'timestamp/service/size'):
                    reader.raw(200, samples())

    def test_timestamp_index_and_selection_validation(self):
        for timestamps in ((100, 100), (200, 100), (100, 201, 300)):
            with self.subTest(timestamps=timestamps), self.assertRaises(ValueError):
                SeedReader(self.paths, timestamps)
        reader = self.reader()
        for selected in ([-1], [PIXELS], [0.5], [[0]], [True], np.array([2 ** 63], dtype=np.uint64)):
            with self.subTest(selected=selected), self.assertRaisesRegex(ValueError, 'Invalid selected'):
                reader.raw(100, selected)

    def test_invalid_record_extents(self):
        for extent in (0, 11, 128 * 1024 * 1024):
            header = bytearray(record(100))
            struct.pack_into('<I', header, 20, extent)
            with self.subTest(extent=extent), self.assertRaisesRegex(ValueError, 'record size'):
                SeedReader._header(header)

    def test_duplicate_and_missing_physical_panels(self):
        duplicate = SeedReader([self.paths[0], self.paths[0]], (100,))
        self.addCleanup(duplicate.close)
        with self.assertRaisesRegex(ValueError, 'ownership'):
            duplicate.raw(100, samples())
        missing = SeedReader([self.paths[0]], (100,))
        self.addCleanup(missing.close)
        with self.assertRaisesRegex(ValueError, 'missing physical panels'):
            missing.raw(100, samples())

    def test_missing_shapes_or_data_and_unsupported_schema(self):
        reader = self.reader()
        for children in (xtc(3), xtc(2, struct.pack('<5I', WORDS, 0, 0, 0, 0))):
            with self.assertRaisesRegex(ValueError, 'ShapesData children'):
                SeedReader._layout(record(100, xtc(1, children, 15)), reader.configs[0])
        schemas = reader.configs[0].config_names()
        for schema in schemas:
            schema['alg_name'] = 'raw'
        with self.assertRaisesRegex(ValueError, 'Configure schema'):
            SeedReader._layout(self.events[0][0], SimpleNamespace(config_names=lambda: schemas))


class CalibrationParityTests(unittest.TestCase):
    def test_native_sampled_calibration_is_bitwise_full_slice(self):
        shape = (32, 2, 5)
        pedestal = np.arange(3 * np.prod(shape), dtype=np.float64).reshape(3, *shape) / 7
        gain = np.full_like(pedestal, 3)
        gain[:, 0, 0, 0] = 0
        offset = np.full_like(pedestal, 0.1)
        constants = {key: (values, {}) for key, values in
                     [('pedestals', pedestal), ('pixel_gain', gain), ('pixel_offset', offset)]}
        raw = (np.arange(np.prod(shape), dtype=np.uint16).reshape(shape) * 113) & 0x3fff
        raw |= (np.indices(shape)[0] % 4).astype(np.uint16) << 14
        selected = np.array([9, 0, 1, 0, 5])
        with patch('reference.SHAPE', shape), patch('reference.PIXELS', 10):
            expected = CpuCalibration(constants)(raw).reshape(32, 10)[:, selected]
            actual = CpuCalibration(constants, selected)(raw.reshape(32, 10)[:, selected])
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


if __name__ == '__main__':
    unittest.main()
