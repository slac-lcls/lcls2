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
class OpalTTFex {
public:
    OpalTTFex(Parameters*);
    ~OpalTTFex();
 public:
    void configure  (XtcData::ConfigIter&,unsigned,unsigned);
    void unconfigure();
    enum TTResult { VALID, NOBEAM, NOLASER, INVALID };
    std::pair<std::vector<double>, TTResult> analyze    (std::vector< XtcData::Array<uint8_t> >& subframes,
                                                         std::vector<double>& sigout,
                                                         std::vector<double>& refout);
 public:
    bool   write_image          () const { return m_prescale_image; }
    bool   write_projections    () const { return m_prescale_projections; }
    bool   write_ref_image      () const { return m_record_ref_image; }
    bool   write_ref_projection () const { return m_record_ref_projection; }
    bool   write_evt_image      ();
    bool   write_evt_projections();
 public:
    std::vector<double>& sig_projection() { return m_sig_avg; }
    std::vector<double>& ref_projection() { return m_ref_avg; }
 public:
  virtual void _monitor_raw_sig (std::vector<double>&);
  virtual void _monitor_ref_sig (std::vector<double>&);
  virtual void _monitor_sub_sig (std::vector<double>&);
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
