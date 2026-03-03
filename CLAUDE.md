# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LCLS II is a scientific data acquisition and analysis framework for the Linear Coherent Light Source (LCLS) at SLAC National Accelerator Laboratory. It consists of four main packages:

- **xtcdata**: Core data format libraries (C++ with CMake)
- **psalg**: Signal processing algorithms and detector calibrations (C++ with Python bindings)  
- **psdaq**: Data acquisition system components (C++ with Python bindings)
- **psana**: Python-based data analysis framework with extensive detector support

## Build System

The project uses a hybrid build system with CMake for C++ components and setuptools for Python packages.

### Environment Setup
Always source the environment before building:
```bash
source setup_env.sh
```

### Building All Packages
```bash
# Build all packages in development mode (most common for development)
./build_all.sh

# Build with specific configuration
./build_all.sh -c Release -p install

# Build without DAQ components (for analysis-only environments)
./build_all.sh -d

# Force clean build (needed when switching between environments)
./build_all.sh -f
```

### Building Individual Packages
```bash
# Build xtcdata only
cd xtcdata && mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$INSTDIR -DCMAKE_PREFIX_PATH=$CONDA_PREFIX ..
make -j 4 install

# Build psana extensions in development mode
cd psana
python setup.py build_ext -f --inplace
pip install --no-deps --prefix=$INSTDIR --editable .
```

### Testing
```bash
# Run psana automated tests
pytest psana/psana/tests/

# Run specific test
pytest psana/psana/tests/test_dgramedit.py
```

## Code Architecture

### Data Flow Architecture
- **XTC Format**: Binary data format defined in xtcdata for high-performance data storage
- **DataSource**: Main entry point in psana for accessing experimental data
- **Detectors**: Modular detector classes in psana/detector/ for different instrument types
- **Event Processing**: Parallel processing via MPI or Legion modes

### Key Components
- `psana/datasource.py`: Primary interface for data access
- `psana/detector/`: Detector-specific analysis modules  
- `psana/smalldata.py`: Reduced data processing for high-throughput analysis
- `psalg/`: Core algorithms for calibration and signal processing
- `xtcdata/xtc/`: Low-level XTC data format implementation

### Python Extension Modules
Many performance-critical components are implemented as Cython extensions:
- `dgramedit.pyx`: Datagram editing and manipulation
- `smdreader.pyx`: Small data reading optimizations
- `parallelreader.pyx`: Parallel data access
- `eventbuilder.pyx`: Event building from multiple data streams

### Environment Variables
- `INSTDIR`: Installation directory (set by build_all.sh)
- `BUILD_LIST`: Subset of extensions to build (e.g., "PEAKFINDER:HEXANODE:CFD")
- `CONDA_PREFIX`: Conda environment path for dependencies
- `TESTRELDIR`: Test release directory for development

## Development Notes

### C++ Code Style  
- Follows Linux kernel brace style with 4-space indentation
- 100-column limit as defined in `.clang-format`
- Never use tabs, always spaces

### Python Package Structure
Each package (psalg, psdaq, psana) follows the same pattern:
- CMakeLists.txt for C++ components
- setup.py for Python extensions and packaging
- pyproject.toml for modern Python build metadata

### MPI Support
psana includes MPI support for parallel processing. When running with multiple cores, global exception handling automatically calls MPI_Abort to prevent hanging processes.

### Development Workflow
The build system supports "develop" mode installations that create symlinks rather than copying files, enabling faster iteration during development.