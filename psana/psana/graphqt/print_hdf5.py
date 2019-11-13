
#----------

import sys
import h5py

#----------

def print_hdf5_file(fname):
    """Prints the HDF5 file structure"""
    f = h5py.File(fname, 'r') # open read-only
    print_hdf5_item(f)
    f.close()
    print('=== EOF ===')

#----------

def print_hdf5_item(g, offset='  ') :
    """Prints the input file/group/dataset (g) name and begin iterations on its content"""

    #print(g.__dir__())
    #print(g.parent)


    if   isinstance(g,h5py.File) :
        print(g.filename, '(File)', g.name)
        print(g.__dir__())
        print(g.id)

    elif isinstance(g,h5py.Dataset) :
        print('(Dataset)', g.name, '    len =', g.shape, g.dtype) #, g.dtype

    elif isinstance(g,h5py.Group) :
        print('(Group)', g.name)

    else :
        print('WORNING: UNKNOWN ITEM IN HDF5 FILE', g.name)
        sys.exit ( "EXECUTION IS TERMINATED" )

    if isinstance(g, (h5py.File, h5py.Group)) :
        for key,val in dict(g).items() :
            subg = val
            print(offset, key, end='') #,"   ", subg.name #, val, subg.len(), type(subg),
            print_hdf5_item(subg, offset + '  ')

#----------

if __name__ == "__main__" :
    print_hdf5_file('/reg/g/psdm/detector/data_test/hdf5/amox27716-r0100-e060000-single-node.h5')
    sys.exit ('End of test')

#----------
