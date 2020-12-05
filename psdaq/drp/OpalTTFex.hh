#pragma once

#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/Array.hh"
#include "xtcdata/xtc/ConfigIter.hh"

#include <vector>
#include <string>

namespace Drp {

class Roi {
public:
  unsigned x0, y0, x1, y1;
};
class Parameters;

class EventInfo {
public:
  bool fixedRate(unsigned rate) const { return _fixedRates & (1<<rate); }
  bool acRate(unsigned rate,
              unsigned tslots) const { return _acRates & (1<<rate) && (_timeSlots&tslots); }
  bool eventCode(unsigned ec) const { return _seqInfo[ec>>4] & (1<<(ec&0xf)); }
  bool sequencer(unsigned word,
                 unsigned bit) const { return _seqInfo[word] & (1<<bit); }
public:
  uint64_t _pulseId; 
  unsigned _fixedRates  : 10;
  unsigned _acRates     : 6;
  unsigned _timeSlots   : 8;
  unsigned _beamPresent : 1, : 3;
  unsigned _beamDestn   : 4;
  uint16_t _seqInfo[18];
};


class OpalTTFex {
public:
    OpalTTFex(Parameters*);
    ~OpalTTFex();
 public:
    void configure  (XtcData::ConfigIter&,unsigned,unsigned);
    void reset      ();
    void unconfigure();
    enum TTResult { VALID, NOBEAM, NOLASER, INVALID };
    TTResult analyze    (std::vector< XtcData::Array<uint8_t> >& subframes);
 public:
    bool   write_image          () const { return m_prescale_image; }
    bool   write_projections    () const { return m_prescale_projections; }
    bool   write_ref_image      () const { return m_record_ref_image; }
    bool   write_ref_projection () const { return m_record_ref_projection; }
    bool   write_evt_image      ();
    bool   write_evt_projections();
 public:
    bool   damaged          () const { return !(m_flt_fwhm>0); }
    double filtered_position() const { return m_flt_position; }
    double filtered_pos_ps  () const { return m_flt_position_ps; }
    double filtered_fwhm    () const { return m_flt_fwhm; }
    double amplitude        () const { return m_amplitude; }
    double next_amplitude   () const { return m_nxt_amplitude; }
    double ref_amplitude    () const { return m_ref_amplitude; }
    std::vector<int>&    sig_projection() { return m_sig; }
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

    std::vector<uint8_t> m_eventcodes_beam_incl;
    std::vector<uint8_t> m_eventcodes_beam_excl;
    std::vector<uint8_t> m_eventcodes_laser_incl;
    std::vector<uint8_t> m_eventcodes_laser_excl;

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

    double   m_ref_convergence;
    double   m_sb_convergence;

    std::vector<double> m_fir_weights;
    std::vector<double> m_calib_poly;

    bool m_ref_empty;
    std::vector<double> m_ref_avg; // accumulated reference
    std::vector<double> m_sb_avg;  // averaged sideband region
    std::vector<int>    m_sig;     // signal region projection
    std::vector<int>    m_sb;      // sideband region
    std::vector<int>    m_ref;     // reference region projection
    unsigned m_pedestal; // from Opal camera configuration

    double m_flt_position;
    double m_flt_position_ps;
    double m_flt_fwhm;
    double m_amplitude;
    double m_nxt_amplitude;
    double m_ref_amplitude;

    std::vector<unsigned> m_cut;
  };

}
