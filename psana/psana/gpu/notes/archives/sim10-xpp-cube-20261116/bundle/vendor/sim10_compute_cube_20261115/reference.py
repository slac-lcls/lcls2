"""Bounded, independent CPU-v3 reference from native seed datagrams."""
import ast
import hashlib
import json
import os
from pathlib import Path
import struct
import time

import numpy as np

SHAPE = (32, 512, 1024)
PIXELS = 512 * 1024
WORDS = 52429
SNAPSHOT = Path('/sdf/data/lcls/ds/prj/prjdatpsana/results/synthetic_mfx/r0001/stage1-20261103')
SAMPLE_SEED = 20261115
MAX_RECORD_BYTES = 128 * 1024 * 1024
MAX_CONTROL_BYTES = 64 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()


def load_constants():
    manifest = json.loads((SNAPSHOT / 'manifest.json').read_text())
    constants, records = {}, {}
    for item in manifest['calibration']:
        document = item['source_document']
        key = document['ctype']
        if item['detector'] != 'jungfrau' or key not in ('pedestals', 'pixel_gain', 'pixel_offset'):
            continue
        array = np.memmap(SNAPSHOT / item['file'], dtype=document['data_dtype'], mode='r',
                          shape=ast.literal_eval(document['data_shape']))
        require(array.shape == (3, *SHAPE) and digest(array) == item['sha256'], 'Frozen constant hash/layout')
        require(key not in constants, 'Duplicate frozen constant')
        constants[key] = (array, document)
        records[key] = dict(shape=list(array.shape), dtype=str(array.dtype),
                            sha256=item['sha256'], bytes=array.nbytes)
    require(len(constants) == 3, 'Missing frozen constants')
    return constants, records


def samples():
    generator = np.random.default_rng(SAMPLE_SEED)
    selected = sorted(set([0, 9, 10, PIXELS - 1] + generator.choice(PIXELS, 12, replace=False).tolist()))
    return np.asarray(selected, dtype=np.int64)


class SeedReader:
    """Read-only, fixed-Configure input; sampled layouts fail closed on change.

    A full native witness per stream supplies zero-copy seed addresses. Only
    Data payloads are omitted from the cached structural witness: every XTC
    header and Shapes byte is compared on EVERY sampled record. Thus extents,
    NamesIds, child order and shapes cannot change unnoticed, even if record
    sizes match. Scalar payloads and non-service environment bits may vary.
    File positions come from walking actual extents, never event-size strides.
    Unbuffered pread requests only structural bytes and unique seed-word runs;
    filesystem page fetches/readahead may still exceed these requested bytes.
    """
    def __init__(self, files, timestamps):
        from psana import dgram
        self.timestamps = tuple(map(int, timestamps))
        require(all(before < after for before, after in zip(self.timestamps, self.timestamps[1:])),
                'Source timestamps must be strictly increasing')
        self.files, self.configs, self.offsets = [], [], []
        self.identities, self.layouts = [], []
        self.seconds = 0.0
        try:
            for filename in files:
                stream = open(filename, 'rb', buffering=0)
                self.files.append(stream)
                identity = self._identity(stream)
                self.identities.append(identity)
                config = self._record(stream, 0)
                require(self._header(config[:24])[1] == 2, 'Source must start with Configure')
                self.configs.append(dgram.Dgram(view=config))
                position, offsets = len(config), {}
                while len(offsets) < len(self.timestamps):
                    header = self._read_at(stream, position, 24)
                    timestamp, service, size = self._header(header)
                    require(position + size <= identity[2], 'Truncated source index record')
                    require(service != 2, 'Source Configure changed')
                    if service == 12:
                        require(timestamp == self.timestamps[len(offsets)], 'Source index event order')
                        offsets[timestamp] = position
                    position += size
                require(self._identity(stream) == identity, 'Source changed during indexing')
                self.offsets.append(offsets)
                self.layouts.append(None)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _identity(stream):
        status = os.fstat(stream.fileno())
        return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns

    @staticmethod
    def _read_at(stream, offset, size):
        payload = os.pread(stream.fileno(), size, offset)
        require(len(payload) == size, 'Short source read')
        return payload

    @staticmethod
    def _header(header):
        require(len(header) == 24, 'Short source header')
        timestamp, environment, _, damage, kind, extent = struct.unpack('<QIIHHI', header)
        require(kind == 0 and damage == 0, 'Invalid source root XTC')
        require(12 <= extent <= MAX_RECORD_BYTES - 12, 'Invalid source record size')
        return timestamp, (environment >> 24) & 15, extent + 12

    @classmethod
    def _record(cls, stream, offset):
        header = cls._read_at(stream, offset, 24)
        _, _, size = cls._header(header)
        require(offset + size <= os.fstat(stream.fileno()).st_size, 'Truncated source datagram')
        return header + cls._read_at(stream, offset + 24, size - 24)

    @staticmethod
    def _layout(payload, config):
        """Preflight the small sim10 schema, then obtain offsets from native views.

        This is deliberately not the GPU descriptor/parser. The native
        Configure supplies Names membership and field sizes; native Dgram
        supplies panel identity and actual array offsets. The Python XTC walk
        only bounds those views and identifies bytes that must remain fixed.
        """
        from psana import dgram
        schemas = {row['names_id_value']: row for row in config.config_names()}
        controls, blocks = [], {}

        def walk(position, limit, allowed, depth=0):
            require(depth <= 16 and position + 12 <= limit, 'Invalid source XTC header')
            source, damage, kind, extent = struct.unpack_from('<IHHI', payload, position)
            stop = position + extent
            require(kind in allowed and damage == 0 and extent >= 12 and stop <= limit,
                    'Invalid source XTC extent/type/damage')
            controls.append((position, payload[position:position + 12]))
            start = position + 12
            if kind in (0, 1):
                children = []
                cursor = start
                while cursor < stop:
                    child_stop, child_kind = walk(cursor, stop, (0, 1) if kind == 0 else (2, 3), depth + 1)
                    children.append((child_kind, cursor + 12, child_stop))
                    cursor = child_stop
                if kind == 1:
                    require(sorted(child[0] for child in children) == [2, 3], 'Invalid ShapesData children')
                    names_id = source & 0x0fffffff
                    require(names_id in schemas and names_id not in blocks, 'Unknown/duplicate source NamesId')
                    schema = schemas[names_id]
                    fields = schema['fields']
                    arrays = [field for field in fields if field['rank'] != 0]
                    require(schema['det_name'] == 'jungfrau' and schema['alg_name'] == 'sim10' and
                            len(arrays) == 1 and arrays[0]['name'] == 'seed' and arrays[0]['rank'] == 1 and
                            arrays[0]['type'] == 1 and arrays[0]['element_size'] == 2,
                            'Unsupported sampled Configure schema')
                    shapes = next(child for child in children if child[0] == 2)
                    data = next(child for child in children if child[0] == 3)
                    require(shapes[2] - shapes[1] == 20 and
                            struct.unpack_from('<I', payload, shapes[1])[0] == WORDS, 'CPU seed shape')
                    data_bytes = WORDS * 2 + sum(field['element_size'] for field in fields if field['rank'] == 0)
                    require(data[2] - data[1] == data_bytes, 'CPU seed Data extent')
                    blocks[names_id] = data[1:]
            elif kind == 2:
                require(stop - start <= MAX_CONTROL_BYTES, 'Oversized source Shapes')
                controls.append((start, payload[start:stop]))
            return stop, kind

        stop, _ = walk(12, len(payload), (0,))
        require(stop == len(payload) and blocks, 'Invalid sampled source tree')
        require(sum(len(content) for _, content in controls) <= MAX_CONTROL_BYTES, 'Oversized source controls')
        decoded = dgram.Dgram(config=config, view=payload)
        buffer = np.frombuffer(payload, np.uint8)
        seeds = {}
        for names_id, (start, stop) in blocks.items():
            panel = schemas[names_id]['segment']
            require(panel not in seeds and 0 <= panel < 32, 'CPU physical panel ownership')
            seed = decoded.jungfrau[panel].sim10.seed
            require(seed.shape == (WORDS,) and seed.dtype == np.uint16 and seed.flags.c_contiguous and
                    np.shares_memory(seed, buffer), 'CPU native seed view')
            offset = seed.ctypes.data - buffer.ctypes.data
            require(start <= offset and offset + seed.nbytes <= stop, 'CPU native seed offset')
            seeds[panel] = offset
        require(set(decoded.jungfrau) == set(seeds), 'CPU native panel membership')
        merged = []
        for offset, content in sorted(controls):
            if merged and merged[-1][0] + len(merged[-1][1]) == offset:
                previous, prefix = merged.pop()
                merged.append((previous, prefix + content))
            else:
                merged.append((offset, content))
        return len(payload), merged, seeds

    def _sampled(self, stream_index, timestamp, selected):
        stream = self.files[stream_index]
        offset = self.offsets[stream_index][timestamp]
        if self.layouts[stream_index] is None:
            payload = self._record(stream, offset)
            require(self._header(payload[:24])[:2] == (timestamp, 12), 'CPU source timestamp/service')
            self.layouts[stream_index] = self._layout(payload, self.configs[stream_index])
        size, controls, seeds = self.layouts[stream_index]
        header = self._read_at(stream, offset, 24)
        require(self._header(header) == (timestamp, 12, size), 'CPU source timestamp/service/size')
        for relative, expected in controls:
            require(self._read_at(stream, offset + relative, len(expected)) == expected,
                    'Sampled source layout changed')
        words, inverse = np.unique(selected // 10, return_inverse=True)
        boundaries = np.flatnonzero(np.diff(words) != 1) + 1
        runs = np.split(np.arange(len(words)), boundaries) if len(words) else []
        result = {}
        for panel, seed_offset in seeds.items():
            values = np.empty(len(words), np.uint16)
            for run in runs:
                content = self._read_at(stream, offset + seed_offset + int(words[run[0]]) * 2, len(run) * 2)
                values[run] = np.frombuffer(content, dtype='<u2')
            result[panel] = values[inverse]
        return result

    def raw(self, timestamp, selected=None):
        from psana import dgram
        begin = time.monotonic()
        timestamp = int(timestamp)
        if selected is not None:
            selected = np.asarray(selected)
            require(selected.ndim == 1 and selected.dtype.kind in 'iu' and
                    np.all((selected >= 0) & (selected < PIXELS)), 'Invalid selected pixels')
            selected = selected.astype(np.int64)
        output = np.empty(SHAPE if selected is None else (32, len(selected)), np.uint16)
        seen = set()
        for stream_index, (stream, config, offsets) in enumerate(zip(self.files, self.configs, self.offsets)):
            require(self._identity(stream) == self.identities[stream_index], 'Source file changed')
            if selected is not None:
                for panel, values in self._sampled(stream_index, timestamp, selected).items():
                    require(panel not in seen, 'CPU physical panel ownership')
                    seen.add(panel)
                    output[panel] = values
                require(self._identity(stream) == self.identities[stream_index], 'Source changed during read')
                continue
            decoded = dgram.Dgram(config=config, view=self._record(stream, offsets[int(timestamp)]))
            require(int(decoded.timestamp()) == timestamp and decoded.service() == 12, 'CPU source timestamp/service')
            for panel, segment in decoded.jungfrau.items():
                require(panel not in seen and 0 <= panel < 32, 'CPU physical panel ownership')
                seen.add(panel)
                seed = segment.sim10.seed
                require(seed.shape == (WORDS,) and seed.dtype == np.uint16, 'CPU seed layout')
                output[panel] = np.repeat(seed, 10)[:PIXELS].reshape(512, 1024)
            require(self._identity(stream) == self.identities[stream_index], 'Source changed during read')
        require(seen == set(range(32)), 'CPU missing physical panels')
        self.seconds += time.monotonic() - begin
        return output

    def close(self):
        for stream in self.files:
            stream.close()


class CpuCalibration:
    """Use the native CPU-v3 kernel with independently packed frozen constants."""
    def __init__(self, constants, selected=None):
        arrays = {key: value[0] for key, value in constants.items()}
        pedestal, gain, offset = (arrays[key] for key in ('pedestals', 'pixel_gain', 'pixel_offset'))
        if selected is not None:
            pedestal, gain, offset = (array.reshape(3, 32, PIXELS)[:, :, selected]
                                      for array in (pedestal, gain, offset))
        combined = pedestal + offset
        reciprocal = np.divide(np.ones_like(pedestal), gain,
                               out=np.zeros(np.broadcast_shapes(pedestal.shape, gain.shape),
                                            dtype=np.result_type(pedestal.dtype, gain.dtype)), where=gain != 0)
        self.shape = combined.shape[1:]
        self.constants = np.zeros((4, int(np.prod(self.shape)), 2), dtype=np.float32)
        for encoded, plane in ((0, 0), (1, 1), (3, 2)):
            self.constants[encoded, :, 0] = combined[plane].reshape(-1)
            self.constants[encoded, :, 1] = reciprocal[plane].reshape(-1)

    def __call__(self, raw):
        import psana.pycalgos.utilsdetector as native
        require(raw.shape == self.shape and raw.dtype == np.uint16, 'CPU calibration raw layout')
        output = np.empty(self.shape, np.float32)
        native.calib_jungfrau_v3(raw, self.constants.reshape(-1), int(np.prod(self.shape[1:])), output.reshape(-1))
        require(np.isfinite(output).all(), 'Nonfinite CPU-v3 result')
        return output


class CubeReference:
    def __init__(self, reader, constants, exhaustive):
        self.reader, self.exhaustive = reader, exhaustive
        self.selected = None if exhaustive else samples()
        self.calibrate = CpuCalibration(constants, self.selected)
        self.records = []

    def __call__(self, bin_index, actual, count):
        begin = time.monotonic()
        expected = np.zeros(self.calibrate.shape, np.float64)
        absolute = np.zeros_like(expected)
        ordinals = range(bin_index, len(self.reader.timestamps), 99) if bin_index < 99 else ()
        checked = 0
        for ordinal in ordinals:
            image = self.calibrate(self.reader.raw(self.reader.timestamps[ordinal], self.selected))
            expected += image
            absolute += np.abs(image.astype(np.float64))
            checked += 1
        require(checked == count, 'CPU bin count')
        observed = actual if self.selected is None else actual.reshape(32, PIXELS)[:, self.selected]
        gamma = count * 2.0 ** -24 / (1 - count * 2.0 ** -24)
        bound = gamma * absolute + 8 * np.finfo(np.float64).eps * absolute + 1e-5
        error = np.abs(observed - expected)
        require(np.isfinite(actual).all() and np.all(error <= bound), 'Cube outside frozen rounding bound')
        if count == 0:
            require(not np.any(actual), 'Empty bin is not zero')
        self.records.append(dict(bin=bin_index, count=count, pixels=int(expected.size),
                                 max_error=float(error.max()), max_bound_ratio=float((error / bound).max()),
                                 pixel_sha256=digest(actual),
                                 seconds=time.monotonic() - begin))
