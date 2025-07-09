""" Returns 

Input: evt, det_no, n_chans
"""

def waveforms(evt, det, n_dets=1, n_chans_per_det=5): 
    """ Returns a dictionary with seg_id as key containing list of 
    waveforms for this segment.
    
    List of segments/ channels:
    MCP, X1, X2, Y1, Y2

    Converts 1D waveform into a list of waveforms using lengths array
    as delimiters.
    """
    waveform_segs = {}
    for segid in range(n_dets * n_chans_per_det):
        wf1d = det._segments(evt)[segid].waveforms
        lengths = det._segments(evt)[segid].lengths
        n_wf = len(lengths)
        waveforms = []
        for i in range(n_wf):
            st = sum(lengths[:i])
            en = st + lengths[i]
            waveforms.append(wf1d[st:en])
        waveform_segs[segid] = waveforms
    return waveform_segs

def times(evt, det, n_dets=1, n_chans_per_det=5):
    startpos_segs = {}
    for segid in range(n_dets * n_chans_per_det):
        startpos_segs[segid] = det._segments(evt)[segid].startpos
    return startpos_segs
