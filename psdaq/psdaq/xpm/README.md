XPM Configuration To Do:

Detector Readout Configuration Parameters

Internal Trigger Delay Range / Units
Absolute Trigger Delay [XPM delay + Internal Trigger Delay]
Internal Event Pipeline Depth : Internal Almost Full Depth
  If no pipeline (Internal Almost Full Depth=0), 
    Minimum Trigger Spacing must cover Upstream RTT
Minimum  Trigger Spacing
  Sets Partition Inhibits
    no more than 1 per Minimum Trigger Spacing
Upstream RTT
  +Internal Almost Full Depth sets Partition Inhibits
    no more than (Almost Full Depth) per Upstream RTT
Downstream RTT 
  +DRP Almost Full Depth sets Partition Inhibits
    no more than (Almost Full Depth) per Downstream RTT

RTT is time from XPM making trigger decision and almost full reaching XPM.
RTT can be reduced by delaying XPM trigger decision, minimizing internal trigger delays.

Teststand Configuration:

Each device gets an IOC to host/apply its configuration
XPM IOC can query IOCs for relevant configuration parameters from well-known fields.
Configure transition launches configuration in execute phase, records results in record phase.  

Who records the XPM or DTI configurations? (A unique DRP process?)

------

Calculating timing parameters for a readout group/partition.

Minimum trigger spacing => XPM Inhibit of no more than 1 trigger per MTS interval
DRP Almost Full Depth   => XPM Inhibit of no more than DRP AFD per longest Downstream RTT
Assuming no internal trigger pipelining => Adjust XPM delay such that Internal Trigger Delay < MTS:
         XPMD = ATD - ITD 
         ITD < min(MTS,ITDmax)  => XPMD > ATD - min(MTS,ITDmax)
         ITD > 0    => XPMD < ATD
         ATD - min(MTS,ITDmax) < XPMD < ATD  for each detector {ATD,ITDmax}

For many detectors:

    max(ATD - min(MTS,ITDmax)) < XPMD < min(ATD)
    
    choose the mean of the range of overlap?  
    XPMD = 0.5*( min(ATD) + max(ATD - min(MTS,ITDmax)) )

    then for each detector:
    ITD = ATD - XPMD

Detectors whose ITD clock = timing clock will always be adjusted by integer steps :)

---

    Downstream RTT needs some knowledge/assumption about readout delay and queueing.  For example, a front end with pipelining will queue the event, and the DRP won't reflect the full status from the new event until that event reaches the front of the queue.  For front ends with no queueing, then only the readout delay needs to be known.  (Should we make use of the L0 Tag to completely track triggers?  DRP can advertise last L0 Tag received + Almost Full Status, so XPM will known how many L0's are still in flight.  Upper bits of full status opcode are the L0 Tag.  This info cannot be summarized by DTI and intermediate XPMs. :(  Since all DRP nodes are the same, then each intermediary can just report the worst case :) )

The DTI allows the upstream queueing and downstream queueing to be tracked independently.  Thus, the DTI knows how many triggers are outstanding in the front end and the almost full status of the backend.  The worst case front end queue depth (or last tag) can be communicated to the XPM; last tag received may be a more accurate indicator.

