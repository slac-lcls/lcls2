# hexanode
A set of modules for python wrapper of the C++ library of RoentDek Hexanode and Quadanode detectors.

## Modules for DLD-MCP calibration and data processing
- ```WFPeaks.py```       - waveform peak finder CFD
- ```WFHDF5IO.py```      - cuastom HDF5 I/O for peak-info
- ```DLDProcessor.py```  - peak-info processing with RoentDek library
- ```DLDStatistics.py``` - accumulator of results in DLDProcessor
- ```DLDGraphics.py```   - drows results accumulated in DLDStatistics

## Examples
```
https://github.com/slac-lcls/lcls2/tree/master/psana/psana/hexanode/examples
psana/psana/hexanode/examples/
  ex-20-data-acqiris-access.py
  ex-21-data-acqiris-graph.py
  ex-22-data-acqiris-peaks-save-h5.py
  ex-23-quad-proc-sort-graph-from-h5.py
  ex-24-quad-proc-sort-graph.py
  ex-25-quad-proc-data.py
```
