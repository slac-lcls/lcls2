# psalg algorithms

## What it is

"psalg" covers two places in the repository:

1. The **psalg package** (`psalg/psalg/`): C++ libraries shared by psana and
   the DAQ (detector geometry, calibration-constant access, HSD digitizer
   decoding, array allocation, the shared-memory transport, logging), plus two
   small Python tools and the Cython module `psalg.utils.dgramCreate`.
2. **Algorithm extensions in psana** (`psana/psana/peakFinder/`,
   `psana/psana/constFracDiscrim/`, `psana/psana/hsd/`, `psana/psana/pycalgos/`):
   C++ algorithms with Cython wrappers that you call from Python, such as
   image peak finders and a constant-fraction discriminator. Note that the
   peak finders live here, not in `psalg/psalg/`.

## Key concepts

### The psalg package (C++)

Subdirectories built by `psalg/psalg/meson.build`:

| Directory | Main classes / functions | Notes |
|---|---|---|
| `geometry/` | `geometry::GeometryAccess`, `GeometryObject`, `SegGeometry*` | Pixel coordinates from a geometry file: `get_pixel_coords(...)`, `get_pixel_coord_indexes(...)`, static `img_from_pixel_arrays(...)`. The Python counterpart is `psana/psana/pscalib/geometry/GeometryAccess.py`. |
| `calib/` | `calib::CalibPars`, `psalg::NDArray`, `MDBWebUtils.hh` | Access to calibration constants (pedestals, gain, status, geometry, ...) and the calibration web service from C++. |
| `detector/` | `detector::Detector`, `detector::AreaDetector` and subclasses (Cspad, Epix100a, Jungfrau, Opal, Pnccd) | C++ detector access on top of `XtcData::ConfigIter`/`DataIter`. |
| `digitizer/` | `Pds::HSD::Channel` (`Hsd.hh`), `Stream.hh` | Decoding of high-speed digitizer (HSD) data; the comments in `Hsd.hh` describe the data format. |
| `alloc/` | `Allocator`, `Stack`, `Heap`, `psalg::AllocArray`, `AllocArray1D` | Header-only allocators and reference-counted arrays used by the algorithms. |
| `shmem/` | `psalg::shmem::ShmemClient`, `psalg::shmem::XtcMonitorServer` | Shared-memory event transport; tools `shmemServer`, `shmemClient`, `shmemWriter` (see [Shared memory and live analysis](shmem-live.md)). |
| `utils/` | `Logger.hh`, `psalg::SysLog`, `DirFileIterator`, `dgramCreate.pyx` | Logging and small utilities, used by the DRP code. `dgramCreate.pyx` is built as the Python module `psalg.utils.dgramCreate`, which builds xtc2 datagrams (imported by `psdaq/psdaq/control/control.py`). |
| `daqPipes/` (Python) | `daqPipes`, `daqStats` commands | Curses displays of DAQ data flow and statistics (`-p/--part`). |

`ShmemClient` (`psalg/psalg/shmem/ShmemClient.hh`) has
`connect(const char* tag, int tr_index=0)`, `get(int& index, size_t& size)` and
`free(int index, size_t size)`; psana wraps it as `PyShmemClient`.

### Algorithm extensions used from Python

Python modules built by `psana/meson.build` (Cython wrappers around C++):

| Python module | Wraps | Python names |
|---|---|---|
| `psana.peakFinder` (re-exports `psana.peakFinder_ext`) | `psana/psana/peakFinder/PeakFinderAlgos.hh`, `LocalExtrema.hh` | class `peak_finder_algos(seg=0, pbits=0, lim_rank=50, lim_peaks=4096)` with `set_peak_selection_parameters(...)` and `peak_finder_v3r3_d2(data, mask, rank, r0, dr, nsigm)` returning `(rows, cols, intens)`; functions such as `local_maxima_1d`, `local_minima_1d`, `threshold_maximums` |
| `psana.psalg_ext` | `PeakFinderAlgosLCLS1.hh` | `peak_finder_algos(seg=0, pbits=0)` with `peak_finder_v3r3_d2` and `peak_finder_v4r3_d2`; used by the Python helpers in `psana/psana/peakFinder/pypsalg.py` (e.g. `peaks_adaptive_2d`, `peaks_droplet_2d`) |
| `psana.peakfinder8` | `psana/psana/peakFinder/peakfinder8.cc` | `peakfinder_8(max_num_peaks, data, mask, pix_r, asic_nx, asic_ny, nasics_x, nasics_y, adc_thresh, hitfinder_min_snr, hitfinder_min_pix_count, hitfinder_max_pix_count, hitfinder_local_bg_radius)` |
| `psana.constFracDiscrim` | `ConstFracDiscrim.hh` | `cfd(sample_interval, horpos, gain, offset, waveform, delay, walk, threshold, fraction)` |
| `psana.hsd` | `psana/psana/hsd/HsdPython.hh`, `psalg/psalg/digitizer/Stream.hh` | the HSD detector interface (`waveforms`, `peaks`, ...; see [Detector interface](detector-interface.md)) |
| `psana.ndarray` | `psana/psana/peakFinder/src/WFAlgos.cc` | `wfpkfinder_cfd(...)` waveform edge finder |
| `psana.utilsdetector_ext` | `psana/psana/pycalgos/UtilsDetector.hh` | `cy_calib_std`, `cy_calib_jungfrau_v0`, ... (used by `psana/psana/detector/UtilsJungfrau.py`) |

These modules are compiled; their signatures above are from the `.pyx`
sources and they do not appear in the generated Python API pages.

The Python class does not mirror the C++ class one to one.
`peak_finder_algos` is defined in `psana/psana/peakFinder/peakFinder_ext.pyx`
(`cdef class peak_finder_algos`, `__cinit__(self, seg=0, pbits=0, lim_rank=50, lim_peaks=4096)`)
and passes all four values to the C++ constructor, so the C++ default
`lim_peaks=6000` shown on the
[psalgos::PeakFinderAlgos](../api/cpp/classpsalgos_1_1PeakFinderAlgos.html)
page does not apply from Python. The Python methods call C++ methods with
other names: `set_peak_selection_parameters` calls `setPeakSelectionPars`,
`peak_finder_v3r3_d2` calls `peakFinderV3r3`, and `print_attributes` calls
`printParameters`.

## Minimal examples

Peak finding on a 2-D array. The calls and parameters are from
`psana/psana/tests/test_psalg.py` (`test_peakFinder`); the test uses a fixed
9x10 float32 array, which is replaced here by a synthetic image with one
bright spot:

```python
import numpy as np
import psana.peakFinder as peakFinder

# Synthetic float32 image: Gaussian noise plus one bright spot at row 25, column 25.
rng = np.random.default_rng(seed=0)
data = rng.normal(loc=20, scale=10, size=(50, 50)).astype(np.float32)
data[24:27, 24:27] += 300
data[25, 25] += 1200

mask = np.ones_like(data, dtype=np.uint16)
pk = peakFinder.peak_finder_algos(pbits=0, lim_peaks=2048)
pk.set_peak_selection_parameters(npix_min=2, npix_max=30, amax_thr=200, atot_thr=600, son_min=7)
rows, cols, intens = pk.peak_finder_v3r3_d2(data, mask, rank=3, r0=4, dr=2, nsigm=0)
# one peak, at about row 25, column 25
```

Constant-fraction discrimination on a waveform, from the same file
(`test_cfd`):

```python
import math
import numpy as np
import psana.constFracDiscrim as cfd

times = np.linspace(0, math.pi, num=10000)
waveform = 10 * np.sin(times)
peak_time = cfd.cfd(math.pi, 0, 10, 0, waveform, 1, 39, 8, 0.5)
# sample_interval, horpos, gain, offset, waveform, delay, walk, threshold, fraction
```

Both run without LCLS data. In an analysis, `data` would come from a detector,
e.g. `det.raw.calib(evt)` for one segment.

## Where in the code

- `psalg/psalg/geometry/`, `calib/`, `detector/`, `digitizer/`, `alloc/`, `shmem/`, `utils/`: C++ libraries.
- `psalg/psalg/daqPipes/`: `daqPipes`, `daqStats`.
- `psana/psana/peakFinder/`: `PeakFinderAlgos.hh`, `PeakFinderAlgosLCLS1.hh`, `LocalExtrema.hh`, `peakfinder8.hh`, `WFAlgos.hh` and the `.pyx` wrappers.
- `psana/psana/constFracDiscrim/`, `psana/psana/hsd/`, `psana/psana/pycalgos/`: other wrapped algorithms.
- `psana/meson.build`: which extension modules are built from which sources.
- Tests: `psana/psana/tests/test_psalg.py`, `psana/psana/tests/test-01-peakfinder.py`, `psalg/psalg/tests/`.

C++ API pages: [geometry::GeometryAccess](../api/cpp/classgeometry_1_1GeometryAccess.html),
[calib::CalibPars](../api/cpp/classcalib_1_1CalibPars.html),
[psalg::shmem::ShmemClient](../api/cpp/classpsalg_1_1shmem_1_1ShmemClient.html),
[psalg::shmem::XtcMonitorServer](../api/cpp/classpsalg_1_1shmem_1_1XtcMonitorServer.html),
[Pds::HSD::Channel](../api/cpp/classPds_1_1HSD_1_1Channel.html),
[psalg::AllocArray](../api/cpp/classpsalg_1_1AllocArray.html),
[psalgos::PeakFinderAlgos](../api/cpp/classpsalgos_1_1PeakFinderAlgos.html).
