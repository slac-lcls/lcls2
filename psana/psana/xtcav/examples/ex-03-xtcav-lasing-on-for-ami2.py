#!/usr/bin/env python
""" Run xtcav lasing on algorithm in external event loop like in AMI. 
"""
#----------

class Arguments:
    fname      = '/reg/g/psdm/detector/data2_test/xtc/data-amox23616-r0137-e000100-xtcav-v2.xtc2'
    experiment = 'amox23616'
    run        = 137
    events     = 50

def test_xtcav_lasing_on(args=Arguments()):

    from psana import DataSource
    from psana.xtcav.LasingOnCharacterization import LasingOnCharacterization, setDetectors
    from psana.pyalgos.generic.NDArrUtils import info_ndarr, np

    ds = DataSource(files=args.fname)
    run = next(ds.runs())

    dets = setDetectors(run)
    lon = LasingOnCharacterization(args, run, dets)

    nimgs=0
    for nev,evt in enumerate(run.events()):

        img = dets._camraw(evt)
        print('Event %03d raw data: %s' % (nev, info_ndarr(img)))

        if img is None: continue
        if not lon.processEvent(evt): continue

        t, power, agr, pulse = lon.resultsProcessImage()
        print('%sAgreement:%7.3f%%  Max power: %g  GW Pulse Delay: %.3f '%(10*' ', agr*100,np.amax(power), pulse[0]))

        nimgs += 1
        if nimgs>=args.events: break

#----------

if __name__ == "__main__":

    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d : %(message)s', datefmt='%Y-%m-%dT%H:%M:%S', level=logging.INFO)

    test_xtcav_lasing_on()

#----------
