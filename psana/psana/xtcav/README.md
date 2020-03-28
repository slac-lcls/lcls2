# Introduction

The XTCAV (X-band Transverse deflecting mode CAVity) is a detector that is used to determine the x-ray power vs. time of each LCLS shot. It can also be used to give information about the relative energies and arrival times of x-ray pulses that are closely spaced in time (fs delays). Unfortunately, some data analysis is necessary to extract this "power profile" from the raw detector signal. This documentation will go through how to do this.

For most experiments, the analysis should be basically turn-key using the code we provide at LCLS. The "Quickstart" section should be enough if you just want to do that as fast as possible. Many experiments that rely heavily on the XTCAV, however, operate in unique electron-beam modes (multibunch, etc) that require more manual fine-tuning of the analysis. To support those cases, we go through a little of the theory behind the XTCAV & analysis routines used to extract the information of interest.

## Documentation

Documentation, including a quickstart guide, is here:
https://confluence.slac.stanford.edu/display/PSDM/New+XTCAV+Documentation

### Prerequisites

This code relies on the psana and PSCalib pacakages, which are automatically available on your SLAC UNIX account. Follow instructions [here](https://confluence.slac.stanford.edu/display/PSDM/psana+python+Setup) to make sure Beyond that, this package uses standard python packages such scipy, numpy, and cv2.


### Installing

If you would like to enhance or change the xtcav code, you can set up the repository locally using the usual 'git clone' method. However, you won't be able to test code on any of the psana data unless you're using one of the psana servers. For development, you should clone the repository into your UNIX account. Once you've ssh-ed into the psana server, in order to use the psana python package, you'll first have to run the command

```
source /reg/g/psdm/etc/psconda.sh
```
`cd` into the `xtcav` directory and run
```
python setup.py develop --user
```

This will allow the changes you make to persist without having to use a make file or reinstall the package. You can then test your changes and subsequently create a pull requests.

### Bugs & desired upgrades

The xtcav code is currently a work in progress. Some features missing include:

* Better electron bunch splitting capabilities
    * The current implementation should theoretically work for any number of bunches in the image. That being said, if bunches heavily intersect, then the code will only find one bunch. Recommendations for solving this include using the current 'connected components' finding method followed by a 'sparsest cut' method that would separate a single region into the two (or more) regions that minimizes the ratio (number of edges cut/min(size of regions)).

* Ability to have different numbers of 'clusters' for different electron bunches within same lasing off reference
    * Most of the groundwork for this has already been laid out. The current issue is saving the lasing off reference when the arrays are of variable length. This is an h5py problem. To fix this, the `FileInterface` file should be changed. Specifically, you'll need to use the special_dtype functionality in h5py to save lists of variable length arrays. The main problem is that it doesn't seem to work for varaible length arrays of arrays...
    Once this has been done, the line `num_groups = num_clusters` in `averageXTCAVProfileGroups` in `Utils.py` simply needs to be removed in order for this functionality to persist. 

* Lasing off profile clustering methods
    * Current clustering algorithms tested include Hierarchical (with cosine, l1, and euclidean distance metrics), KMeans, DBSCAN, and Birch. Analysis showed that Hierarchical with Euclidean affinity provided the best results. See child page of XTCAV confluence for specific results. The next step would be to compare these algorithms with the performance of a SVD composition method: <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4052871/>

* xtcavDisplay script
    * The scripts in `bin`, specifically `xtcavDisp`, are currently not very customizeable. Some desired features include providing flags to change the types of graphs shown, the size of graphs, etc. This would make it more useful for realtime anayses. 
    
* A common GUI that could be used in the hutches and in ACR. Ideally it could be run online and offline. This is challenging due to the way XTCAV data travel over the BLD to the hutches.
    * remote desktop to ACR
    * send image at 1 Hz via EPICS —> psana
    * ~1 min delayed SMD (requires DAQ)
    * ratner: rip out algorithm to TREX, call python underneath

* Add default values for xtcav global calibration values
    * The calibration values for the xtcav are currently populated from the psana datasource. It would be good to have some default values for variables such as umperpix, strstrength, etc. in case that information is missing or for running siumulations/experiments. Default values can be added to the `GlobalCalibration` namedtuple in Utils.py

* Variability in optimal number of clusters chosen
    * Because the reference sets used to calculate the gap statistic are generated randomly, there may be some variability in the number of clusters chosen. Therefore your results may be slightly different from run to run. If you’d like to avoid this, you can manually set the number of clusters chosen
    
And a few small things
* Pixel percentage: make it a parameter
* Handle no ROI Info

### Data Used to Develop/Debug the Code

* amox23616 — SASE 
    * 104 dark, 131 las off, 132-139 data
    * 12 dark, 60 las off, 62-63 data
* cxin7316 — clemens [not used]
* diamcc14 — 441 dark, 442 las off, 443 data

