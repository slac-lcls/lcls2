
#----------

import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import find_peaks

#----------

def peak_finder_v2(wf, sigmabins, threshold, deadbins, pkvals, pkinds) :
    """ v2 peak-finder:
        - waveform wf (1-d np.array) is convolved with gaussian(sigmabins),
        - intensiies above exceeding threshold are considered only,
        - local max in intensity are found and returned as peaks.
    """
    # gaussian_filter1d - LPF
    wff = gaussian_filter1d(wf, sigmabins, axis=-1, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)

    # apply threshold to intensity
    wffs = np.select([wff<threshold,], [wff,], default=0)

    # find_peaks - find local max in intensity (sign "-" inverts intensity)
    pinds, _ = find_peaks(-wffs, height=None, distance=deadbins)

    NUM_HITS = pkinds.size
    npeaks = pinds.size if pinds.size<=NUM_HITS else NUM_HITS 
    pkinds[:npeaks] = pinds[:npeaks]
    pkvals[:npeaks] = wf[pinds[:npeaks]]

    return npeaks

#----------

def peak_finder_v3(wf, sigmabins, basebins, nstdthr, gapbins, deadbins, pkvals, pkinds) :
    """ v3 peak-finder:
        - process waveform with bpf
        - local max in intensity are found and returned as peaks.
    """
    # apply bpf filter
    wfgi, wff, wfg, thrg, edges = bpf(wf, sigmabins, basebins, nstdthr, gapbins)

    # find_peaks - find local max in intensity (sign "-" inverts intensity)
    pinds, _ = find_peaks(-wfgi, distance=deadbins)

    print('XXX peak_finder_v3 pinds', pinds)

    NUM_HITS = pkinds.size
    npeaks = pinds.size if pinds.size<=NUM_HITS else NUM_HITS 
    pkinds[:npeaks] = pinds[:npeaks]
    pkvals[:npeaks] = wf[pinds[:npeaks]]

    return npeaks, wfgi, wff, wfg, thrg, edges

#----------

def bpf(wf, sigmabins=3, basebins=100, nstdthr=5, gapbins=100) :
    """ Band path filter for 1-d waveform 
        - eliminates low and high friquencies in the waveform.
    """
    wff = gaussian_filter1d(wf, sigmabins, axis=-1, order=0,\
                 output=None, mode='reflect', cval=0.0, truncate=4.0)
    wfg = np.gradient(wff)
    std_wfg = wfg[:basebins].std()
    thrg = std_wfg*nstdthr
    cond_sig_wfg = np.absolute(wfg)>thrg
    sig_wfg = np.select([cond_sig_wfg,], [1,], default=0)
    inds = np.nonzero(sig_wfg)[0] # index selects 0-s dimension in tuple
    grinds = np.split(inds, np.where(np.diff(inds)>gapbins)[0]+1)
    edges = [(g[0],g[-1]) for g in grinds]
    wfgi = np.zeros_like(wfg)
    for (b,e) in edges : wfgi[b:e] = np.cumsum(wfg[b:e])

    return wfgi, wff, wfg, thrg, edges

#----------

def wavelet(x, scx=0.5, xoff=-2.4, trise=11, tdecr=40, n=5) :
    a = 1./trise
    g = 1./tdecr
    b = a + g
    B = -(a+b)*n/(2*a*b*scx)
    C = n*(n-1)/(a*b*scx*scx)
    xpk = -B-sqrt(B*B-C)
    x0 = xpk*scx
    norm = pow(x0,n-1)*(n-a*x0)*exp(-b*x0)
    xc = (x+xpk+xoff)*scx
    return -np.power(xc,n-1)*(n-a*xc)*exp_desc_step(b*xc) / norm

#----------

def split_consecutive(arr, gap=1):
    """spit array of indexes for arrays of consequtive groups of indexes: returns list of arrays"""
    return np.split(arr, np.where(np.diff(arr)>gap)[0]+1)

#----------
#----------
#----------
#----------
#----------
#----------
#----------
#----------
