from libcpp.vector cimport vector

ctypedef vector[double] Waveform

cdef extern from "psalg/constFracDiscrim/ConstFracDiscrim.hh" namespace "psalgos":
    double getcfd(const double sampleInterval,
                  const double horpos,
                  const double gain,
                  const double offset,
                  const Waveform &waveform,
                  const signed int delay,
                  const double walk,
                  const double threshold,
                  const double fraction)


def cfd(sample_interval, horpos, gain, offset, waveform, delay, walk, threshold, fraction):
    return getcfd(sample_interval, horpos, gain, offset, waveform, delay, walk, threshold, fraction)
