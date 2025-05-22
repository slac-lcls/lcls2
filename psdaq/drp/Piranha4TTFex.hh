#pragma once

#include "TTFex.hh"
#include "psdaq/service/Semaphore.hh"

#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/Array.hh"
#include "xtcdata/xtc/ConfigIter.hh"

#include "psalg/calib/NDArray.hh"

#include <vector>
#include <string>

namespace Drp {
class Parameters;
class Piranha4TTFex {
public:
    Piranha4TTFex(Parameters*);
    ~Piranha4TTFex();
 public:
    void configure  (XtcData::ConfigIter&,unsigned);
    void reset      ();
    void unconfigure();
    enum TTResult { VALID, NOBEAM, NOLASER, INVALID };
    TTResult analyze    (std::vector< XtcData::Array<uint8_t> >& subframes,
                         std::vector<double>& sigout,
                         std::vector<double>& refout);
 public:
    bool   write_image       () const { return m_prescale_image; }
    bool   write_averages    () const { return m_prescale_averages; }
    bool   write_ref_image   () const { return m_record_ref_image; }
    bool   write_ref_average () const { return m_record_ref_average; }
    bool   write_evt_image   ();
    bool   write_evt_averages();
 public:
    bool   damaged          () const { return !(m_flt_fwhm>0); }
    double filtered_position() const { return m_flt_position; }
    double filtered_pos_ps  () const { return m_flt_position_ps; }
    double filtered_fwhm    () const { return m_flt_fwhm; }
    double amplitude        () const { return m_amplitude; }
    double next_amplitude   () const { return m_nxt_amplitude; }
    double ref_amplitude    () const { return m_ref_amplitude; }
    std::vector<double>& sig_average() { return m_sig_avg; }
    std::vector<double>& ref_average() { return m_ref_avg; }
 public:
  virtual void _monitor_raw_sig (std::vector<double>&);
  virtual void _monitor_ref_sig (std::vector<double>&);
  virtual void _monitor_sub_sig (std::vector<double>&);
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

    double m_flt_position;
    double m_flt_position_ps;
    double m_flt_fwhm;
    double m_amplitude;
    double m_nxt_amplitude;
    double m_ref_amplitude;

    std::vector<unsigned> m_cut;
  };

}
