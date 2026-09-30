"""Run a copied external user kernel through public serial/MPI APIs only."""
import argparse
from contextlib import closing
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np


def write_cpu_reference(directory):
    """Independent CPU-only DataSource; use actual default detector calibration."""
    from psana import DataSource
    assert directory is not None
    directory.mkdir(exist_ok=False)
    manifest = dict(events={}, constants={})
    ds = DataSource(exp='mfx100848724', run=51, dir='/sdf/data/lcls/ds/prj/public01/xtc',
                    detectors=['jungfrau'], max_events=13)
    for run in ds.runs():
        det = run.Detector('jungfrau')
        with closing(run.events()) as events:
            for event in events:
                raw = det.raw.raw(event)
                image = det.raw.calib(event)  # No mask/offset/version overrides.
                assert raw.shape == (32, 512, 1024), raw.shape
                image = np.asarray(image).reshape(raw.shape)
                assert image.dtype == np.float32 and np.any(image != 0)
                stamp = str(int(event.timestamp))
                assert stamp not in manifest['events']
                filename = stamp + '.npy'
                np.save(directory/filename, image)
                manifest['events'][stamp] = dict(file=filename,
                    sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                    raw_sha256=[hashlib.sha256(panel.tobytes()).hexdigest() for panel in raw])
        manifest['panels'] = 32
        for key in ('pedestals', 'pixel_gain', 'pixel_offset', 'pixel_status', 'status_extra'):
            entry = det.calibconst.get(key)
            if entry is not None and entry[0] is not None:
                value = entry[0]
                manifest['constants'][key] = dict(shape=list(value.shape), dtype=str(value.dtype),
                    sha256=hashlib.sha256(value.tobytes()).hexdigest())
    assert len(manifest['events']) == 13
    assert 'cupy' not in sys.modules
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('CPU_CALIB_REFERENCE_OK ' + json.dumps(dict(events=13, constants=manifest['constants'])), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--user-dir', type=Path)
    p.add_argument('--mode', choices=('cpu', 'serial', 'exclusive', 'hybrid'), required=True)
    p.add_argument('--offset', action='store_true')
    p.add_argument('--status-bits', type=lambda value: int(value, 0), default=0)
    p.add_argument('--pinned-bytes', type=int, default=64 << 20)
    p.add_argument('--cpu-reference', type=Path, help='CPU output directory, created by --mode cpu')
    a = p.parse_args()
    if a.mode == 'cpu':
        write_cpu_reference(a.cpu_reference)
        return
    if a.user_dir is None:
        p.error('--user-dir is required for GPU modes')
    reference = None
    if a.cpu_reference:
        reference = json.loads((a.cpu_reference/'manifest.json').read_text())
        a.offset = 'pixel_offset' in reference['constants']
        a.status_bits = (1 << 64) - 1 if 'pixel_status' in reference['constants'] else 0
    extra_bits = (1 << 64) - 1 if reference and 'status_extra' in reference['constants'] else 0
    sys.path.insert(0, str(a.user_dir.resolve()))
    user_module = importlib.import_module('jungfrau_calibration')
    assert Path(user_module.__file__).resolve().parent == a.user_dir.resolve()
    from psana import DataSource
    from psana.gpu import GpuTask
    from psana.tests.gpu.user_calibration_reference import calibrate
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    assert comm.size == (1 if a.mode == 'serial' else 4)
    user = user_module.JungfrauCalibration(use_offset=a.offset, status_bits=a.status_bits,
                                          stextra_bits=extra_bits)
    assert 'cupy' not in sys.modules
    batches, seen, retained = [], [], []
    launches = []
    def callback(batch, stream):
        import cupy as cp
        # Instrument the real RawKernel submissions, outside the user's module.
        original = cp.RawKernel
        def counted(*args, **kwargs):
            kernel = original(*args, **kwargs)
            def launch(grid, block, argv, **kw):
                launches.append((batch_size_from_total(argv), kw['stream'].ptr))
                return kernel(grid, block, argv, **kw)
            return launch
        # Capture scalar metadata, never a callback-scoped context in cached kernels.
        segment_count = len(batch.segment_ids('jungfrau'))
        def batch_size_from_total(argv):
            return int(argv[7]) // (int(argv[8]) * segment_count)
        cp.RawKernel = counted
        try:
            user(batch, stream)
        finally:
            cp.RawKernel = original
        batches.append(batch.size)
    task = GpuTask(callback, inputs=user.inputs, calibconst=user.calibconst)
    options = dict(exp='mfx100848724', run=51, dir='/sdf/data/lcls/ds/prj/public01/xtc',
                   detectors=['jungfrau'], gpu_fn=task, max_events=13,
                   batch_size=5, n_gpu_streams=2, gpu_bulk_read=True,
                   gpu_d2h_pinned_bytes=a.pinned_bytes)
    options['hybrid_det' if a.mode == 'hybrid' else 'gpu_det'] = 'jungfrau'
    for run in DataSource(**options).runs():
        det = run.Detector('jungfrau')
        with closing(run.events()) as events:
            for event in events:
                fields = event.gpu.detector('jungfrau').field('raw', 'raw').on_cpu
                segments = tuple(fields.segment_ids)
                panels = []
                for segment in segments:
                    panel = fields[segment]
                    if panel.ndim == 3 and panel.shape[0] == 1:
                        panel = panel[0]
                    assert panel.ndim == 2, panel.shape
                    panels.append(panel)
                raw = np.stack(panels)[None]
                if reference:
                    # Full public Jungfrau fixture: CPU raw rows are physical IDs 0..31.
                    assert sorted(segments) == list(range(reference['panels']))
                    item = reference['events'][str(int(event.timestamp))]
                    for segment, panel in zip(segments, panels):
                        assert hashlib.sha256(panel.tobytes()).hexdigest() == item['raw_sha256'][segment]
                    expected = np.load(a.cpu_reference/item['file'], mmap_mode='r')[list(segments)]
                else:
                    constants = {k: det.calibconst[k][0] for _, k in user.calibconst}
                    expected = calibrate(raw, np.ones(raw.shape[:2], np.uint8), segments,
                                         constants, use_offset=a.offset, status_bits=a.status_bits)[0]
                result = event.gpu.get(user.output)
                actual = result.on_cpu
                np.testing.assert_array_equal(actual, expected)
                digest = hashlib.sha256(actual.tobytes()).hexdigest()
                seen.append((int(event.timestamp), digest))
                if not retained:
                    retained.append((result, digest))
    for result, digest in retained:
        assert hashlib.sha256(result.on_cpu.tobytes()).hexdigest() == digest
    assert user.calls == len(batches) == len(launches)
    assert [n for n, _ in launches] == batches
    if not batches:
        assert 'cupy' not in sys.modules
    rows = comm.gather(dict(rank=comm.rank, seen=seen, batches=batches,
                           launches=len(launches)), root=0)
    if comm.rank == 0:
        stamps = [ts for r in rows for ts, _ in r['seen']]
        sizes = [n for r in rows for n in r['batches']]
        assert len(stamps) == len(set(stamps)) == 13
        assert sorted(sizes) == [3, 5, 5]
        print('USER_CALIBRATION_OK ' + json.dumps(dict(mode=a.mode, offset=a.offset,
            status_bits=a.status_bits, stextra_bits=extra_bits,
            reference='det.raw.calib(evt)' if reference else 'numpy-policy', pinned_bytes=a.pinned_bytes, ranks=rows)), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        from mpi4py import MPI
        if MPI.COMM_WORLD.size > 1:
            MPI.COMM_WORLD.Abort(1)
        raise
