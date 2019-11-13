#!/usr/bin/env python
#----------
"""1. opens hdf5 with processed waveform created by the script like psana/hexanode/examples/ex-12-quad-proc-data-save-h5.py,
   2. loop over events in hdf5 file using iterator f.next_event(),
   3. accesses and prints data from hdf5 file.
"""
#----------

import logging
logger = logging.getLogger(__name__)

from psana.hexanode.WFHDF5IO import open_input_h5file
from psana.pyalgos.generic.NDArrUtils import print_ndarr

#----------

def test_data_from_hdf5(**kwargs):

    IFNAME = kwargs['ifname']
    EVSKIP = kwargs['evskip']
    EVENTS = kwargs['events'] + EVSKIP

    f = open_input_h5file(IFNAME, **kwargs)

    print('  file: %s\n  number of events in file %d' % (IFNAME, f.events_in_h5file()))

    while f.next_event() :
        evnum = f.event_number()

        if evnum<EVSKIP : continue
        if evnum>EVENTS : break
        print('Event %3d'%evnum)
        print_ndarr(f.number_of_hits(), '  number_of_hits:')
        print_ndarr(f.tdc_ns(),         '  peak_times_ns: ', last=4)

#----------

if __name__ == "__main__" :

    import sys
    logging.basicConfig(format='%(levelname)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S', level=logging.DEBUG)

    print(50*'_')
    tname = sys.argv[1] if len(sys.argv) > 1 else '1'
    print('TEST %s' % tname)

    kwargs = {'ifname' : '/reg/g/psdm/detector/data_test/hdf5/amox27716-r0100-e060000-single-node.h5',
              'evskip' : 7,
              'events' : 10,
             }\
             if tname == '1' else\
             {'ifname' : 'amox27716-r0100-e000005-single-node.h5',
              'evskip' : 0,
              'events' : 5,
             }

    test_data_from_hdf5(**kwargs)

    sys.exit('END OF %s %s' % (sys.argv[0], tname))

#----------
