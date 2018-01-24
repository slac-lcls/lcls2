
# Creating a psconda environment in RHEL6 with the latest h5py


ssh psbuild-rhel6

conda create -n snake_rhel6 --clone ana-1.3.44

source activate snake_rhel6

conda install -c anaconda hdf5