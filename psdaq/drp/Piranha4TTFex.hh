/**
 * @file
 * @brief Piranha4TTFex, timetool edge finding on Piranha4 line-camera data.
 */
#pragma once

#include "TTFex.hh"
#include "psdaq/service/Semaphore.hh"

#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/Array.hh"
#include "xtcdata/xtc/ConfigIter.hh"

#include "psalg/calib/NDArray.hh"

#include <string>
#include <vector>
#include <utility>

namespace Drp {
class Parameters;
/** Timetool feature extraction for Piranha4 line data: extracts the signal region of interest, divides the averaged signal by a rolling reference, filters it and fits the edge. The reference is loaded from and saved to a file. */
class Piranha4TTFex {
public:
    /** Load the reference from the file named by the ttreffile kwarg (default detName.ttref; relative names are taken under $HOME, or /tmp if HOME is unset), if that file exists. */
    Piranha4TTFex(Parameters*);
    /** Save the accumulated reference to the reference file. */
    ~Piranha4TTFex();
 public:
    /** Read the fex settings from the configuration (FIR weights, optionally inverted, calibration polynomial, beam and laser event selections, signal minimum, prescales, convergence factors, pedestal from user.black_level minus fex.pedestal_adj, signal region and reference record mode) for a line whose pixel count is the unsigned argument, and reset the counters. */
    void configure  (XtcData::ConfigIter&,unsigned);
    /** Print the fraction of calls cut at each stage, if analyze() was called. */
    void unconfigure();
    /** Outcome of analyze(); for results other than VALID the negated value is also stored in the amplitude slot. */
    enum TTResult { VALID, /**< An edge was fitted (0). */ NOBEAM, /**< Beam not selected; the reference was updated instead (1). */ NOLASER, /**< Laser not selected; nothing else was done (2). */ INVALID  /**< A frame, signal, reference, peak or width check failed (3). */ };
    /** Analyze one event: subframes[3] holds the EventInfo and subframes[2] the pixel line. Returns six values (filtered position, calibrated position, amplitude, next amplitude, reference amplitude, FWHM) with a TTResult; sigout receives the signal (divided by the reference if that stage is reached) and refout a copy of the reference average when it is used. */
    std::pair<std::vector<double>, TTResult> analyze    (std::vector< XtcData::Array<uint8_t> >& subframes,
                                                         std::vector<double>& sigout,
                                                         std::vector<double>& refout);
 public:
    /** Return true if the image prescale (fex.prescale.image) is non-zero. */
    bool   write_image       () const { return m_prescale_image; }
    /** Return true if the averages prescale (fex.prescale.averages) is non-zero. */
    bool   write_averages    () const { return m_prescale_averages; }
    /** Return true if the reference record mode asks for reference images (mode 2). */
    bool   write_ref_image   () const { return m_record_ref_image; }
    /** Return true if the reference record mode asks for reference averages (mode 1). */
    bool   write_ref_average () const { return m_record_ref_average; }
    /** Always returns true: the function returns before its prescale logic, which is dead code. */
    bool   write_evt_image   ();
    /** Always returns false: the function returns before its prescale logic, which is dead code. */
    bool   write_evt_averages();
 public:
    /** Return the rolling average of the signal. */
    std::vector<double>& sig_average() { return m_sig_avg; }
    /** Return the rolling average of the reference. */
    std::vector<double>& ref_average() { return m_ref_avg; }
 public:
  /** Hook called with the raw signal; prints samples only when built with DBUG2. */
  virtual void _monitor_raw_sig (std::vector<double>&);
  /** Hook called with the reference signal; prints samples only when built with DBUG2. */
  virtual void _monitor_ref_sig (std::vector<double>&);
  /** Hook called with the reference-normalized signal; prints samples only when built with DBUG2. */
  virtual void _monitor_sub_sig (std::vector<double>&);
  /** Hook called with the filtered signal; prints samples only when built with DBUG2. */
  virtual void _monitor_flt_sig (std::vector<double>&);
private:
    std::string m_fname;

    unsigned m_pixels;

    EventSelect m_beam_select;
    EventSelect m_laser_select;

    int      m_signal_minvalue;  // valid signal must be at least this large

    //    int      m_subtractAndNormalize;

    Roi m_sig_roi;

    unsigned m_prescale_image;
    unsigned m_prescale_averages;
    unsigned m_prescale_image_counter;
    unsigned m_prescale_averages_counter;

    bool     m_record_ref_image;
    bool     m_record_ref_average;

    double   m_sig_convergence;
    double   m_ref_convergence;

    std::vector<double> m_fir_weights;
    std::vector<double> m_calib_poly;

    bool m_ref_empty;
    Pds::Semaphore m_sig_avg_sem;
    std::vector<double> m_sig_avg; // accumulated signal
    Pds::Semaphore m_ref_avg_sem;
    std::vector<double> m_ref_avg; // accumulated reference
    int m_pedestal; // from Piranha4 camera configuration

    std::vector<unsigned> m_cut;
  };

}
