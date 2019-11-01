from psana.pyalgos.generic.NDArrUtils import print_ndarr

from psana import DataSource
ds = DataSource(files='/reg/g/psdm/detector/data2_test/xtc/data-amox27716-r0100-acqiris-e000100.xtc2')
myrun = next(ds.runs())
for nevt,evt in enumerate(myrun.events()):
    if nevt>10 : break
    print('Event %d'%nevt)
    #print(evt._dgrams[0].tmo_hexanode[0].raw.times.shape)
    #print(evt._dgrams[0].tmo_hexanode[0].raw.waveforms.shape)
    print_ndarr(evt._dgrams[0].tmo_hexanode[0].raw.times, '  times : ')
    print_ndarr(evt._dgrams[0].tmo_hexanode[0].raw.waveforms, '  wforms: ')
    # run findEdges here
    # call resort64 here
