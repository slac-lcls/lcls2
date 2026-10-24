# lcls2

lcls2 is the software repository for data acquisition and data analysis at
LCLS-II (SLAC). It holds four packages that are built together by the root
`meson.build`:

| Package | Language | What it is |
|---|---|---|
| **psana** (`psana/`) | Python, Cython, C++ | psana2, the analysis framework: `DataSource`, runs, events, detector interfaces, calibration, MPI and shared-memory modes. |
| **psdaq** (`psdaq/`) | C++, Python | The DAQ: control (`psdaq/psdaq/control`), data reduction pipeline (`psdaq/drp`), event builders (`psdaq/psdaq/eb`), hardware tools. |
| **xtcdata** (`xtcdata/`) | C++ | The XTC2 data format library (`xtcdata/xtcdata/xtc`) and command-line tools such as `xtcreader`. |
| **psalg** (`psalg/`) | C++, Python | Support libraries used by psana and the DAQ: detector geometry, calibration-constant access, HSD digitizer processing, array allocation, shared-memory transport (`psalg/psalg/shmem`), logging utilities. |

## How these docs are organized

- [Getting started](getting-started.md): build and environment pointers taken from
  `README.md` and the build scripts.
- [C++ API](api/cpp/index.html): Doxygen pages for the public headers of
  xtcdata, psalg, psdaq and psana. The search box of this site does not cover
  them; the C++ pages have their own search box.

The site is versioned with [mike](https://github.com/jimporter/mike): `dev`
follows the `master` branch of
[slac-lcls/lcls2](https://github.com/slac-lcls/lcls2), and release tags
(such as `4.3.4`) get their own version. The version selector is next to the
site title.
