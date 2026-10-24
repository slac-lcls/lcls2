/**
 * @file
 * @brief OpalTTFex, timetool edge finding on Opal camera images.
 */
#pragma once

#include "TTFex.hh"
#include "psdaq/service/Semaphore.hh"

#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/Array.hh"
#include "xtcdata/xtc/ConfigIter.hh"

#include "psalg/calib/NDArray.hh"

#include <string>
#include <utility>
#include <vector>

namespace Drp {
class Parameters;
/** Timetool feature extraction for Opal camera images: projects the signal (and optional reference and sideband) regions of interest, divides the averaged signal by a rolling reference, filters it and fits the edge. The reference is loaded from and saved to a file. */
class OpalTTFex {
public:
    /** Load the reference projection from the file named by the ttreffile kwarg (default detName.ttref; relative names are taken under $HOME, or /tmp if HOME is unset), if that file exists. */
    OpalTTFex(Parameters*);
    /** Save the accumulated reference projection to the reference file. */
    ~OpalTTFex();
 public:
    /** Read the fex settings from the configuration (FIR weights, optionally inverted, calibration polynomial, beam and laser event selections, projection axis and minimum, prescales, convergence factors, pedestal from user.black_level minus fex.pedestal_adj, regions of interest and reference record mode) for an image whose columns and rows are the two unsigned arguments, and reset the counters. */
    void configure  (XtcData::ConfigIter&,unsigned,unsigned);
    /** Print the fraction of calls cut at each stage, if analyze() was called. */
    void unconfigure();
    /** Outcome of analyze(); for results other than VALID the negated value is also stored in the amplitude slot. */
    enum TTResult { VALID, /**< An edge was fitted (0). */ NOBEAM, /**< Beam not selected; the reference was updated instead (1). */ NOLASER, /**< Laser not selected; nothing else was done (2). */ INVALID  /**< A frame, projection, reference, peak or width check failed (3). */ };
    /** Analyze one event: subframes[3] holds the EventInfo and subframes[2] the image. Returns six values (filtered position, calibrated position, amplitude, next amplitude, reference amplitude, FWHM) with a TTResult; sigout receives the signal projection (divided by the reference if that stage is reached) and refout a copy of the reference average when it is used. */
    std::pair<std::vector<double>, TTResult> analyze    (std::vector< XtcData::Array<uint8_t> >& subframes,
                                                         std::vector<double>& sigout,
                                                         std::vector<double>& refout);
 public:
    /** Return true if the image prescale (fex.prescale.image) is non-zero. */
    bool   write_image          () const { return m_prescale_image; }
    /** Return true if the projection prescale (fex.prescale.projections) is non-zero. */
    bool   write_projections    () const { return m_prescale_projections; }
    /** Return true if the reference record mode asks for reference images (mode 2). */
    bool   write_ref_image      () const { return m_record_ref_image; }
    /** Return true if the reference record mode asks for reference projections (mode 1). */
    bool   write_ref_projection () const { return m_record_ref_projection; }
    /** Return true, and restart the count, once the number of images analyzed since the last true reaches the image prescale. */
    bool   write_evt_image      ();
    /** Return true, and restart the count, once the number of projections made since the last true reaches the projection prescale. */
    bool   write_evt_projections();
 public:
    /** Return the rolling average of the signal projection. */
    std::vector<double>& sig_projection() { return m_sig_avg; }
    /** Return the rolling average of the reference projection. */
    std::vector<double>& ref_projection() { return m_ref_avg; }
 public:
  /** Hook called with the raw signal projection; prints samples only when built with DBUG2. */
  virtual void _monitor_raw_sig (std::vector<double>&);
  /** Hook called with the reference projection; prints samples only when built with DBUG2. */
  virtual void _monitor_ref_sig (std::vector<double>&);
  /** Hook called with the reference-normalized signal; prints samples only when built with DBUG2. */
  virtual void _monitor_sub_sig (std::vector<double>&);
  /** Hook called with the filtered signal; prints samples only when built with DBUG2. */
  virtual void _monitor_flt_sig (std::vector<double>&);
private:
    std::string m_fname;

    unsigned m_columns;
    unsigned m_rows;

    EventSelect m_beam_select;
    EventSelect m_laser_select;

    unsigned m_project_axis    ;  // project image onto Y axis
    int      m_project_minvalue;  // valid projection must be at least this large

    //    int      m_subtractAndNormalize;

    unsigned m_use_ref_roi;
    unsigned m_use_sb_roi;
    Roi m_sig_roi, m_sb_roi, m_ref_roi;

    unsigned m_prescale_image;
    unsigned m_prescale_projections;
    unsigned m_prescale_image_counter;
    unsigned m_prescale_projections_counter;

    bool     m_record_ref_image;
    bool     m_record_ref_projection;

    double   m_sig_convergence;
    double   m_ref_convergence;
    double   m_sb_convergence;

    std::vector<double> m_fir_weights;
    std::vector<double> m_calib_poly;

    bool m_ref_empty;
    Pds::Semaphore m_sig_avg_sem;
    std::vector<double> m_sig_avg; // accumulated signal
    Pds::Semaphore m_ref_avg_sem;
    std::vector<double> m_ref_avg; // accumulated reference
    Pds::Semaphore m_sb_avg_sem;
    std::vector<double> m_sb_avg;  // averaged sideband region
    unsigned m_pedestal; // from Opal camera configuration

    std::vector<unsigned> m_cut;
  };

}
