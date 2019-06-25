#include <unistd.h>

#include "xtcdata/xtc/Dgram.hh"

class Configuration {
public:
public:
  Configuration(unsigned raw_gate_start,
                unsigned raw_gate_rows,
                unsigned fex_gate_start,
                unsigned fex_gate_rows,
                unsigned fex_thresh_lo,
                unsigned fex_thresh_hi) :
    _raw_gate_start (raw_gate_start),
    _raw_gate_rows  (raw_gate_rows ),
    _fex_gate_start (fex_gate_start),
    _fex_gate_rows  (fex_gate_rows ),
    _fex_thresh_lo  (fex_thresh_lo ),
    _fex_thresh_hi  (fex_thresh_hi )
  {}
public:
  unsigned _raw_gate_start;
  unsigned _raw_gate_rows;
  unsigned _fex_gate_start;
  unsigned _fex_gate_rows;
  unsigned _fex_thresh_lo;
  unsigned _fex_thresh_hi;
};

class StreamHeader {
public:
  StreamHeader() {}
public:
  //  unsigned num_samples() const { return _p[0]&~(1<<31); }
  unsigned num_samples() const { 
    unsigned v = _p[0]&~(1<<31);
    //    if (!v) v = 0x640;  // kludge for fw bug
    //    if (!v) v = 0x320;  // kludge for fw bug
    return v;
  }
  unsigned stream_id  () const { return (_p[1]>>24)&0xff; }
  unsigned cache_len  (unsigned row_size) const { 
    return (((_p[3]>>20) - (_p[3]>>4))&0xfff)*row_size + 
      ((_p[3]>>16)&0xf) - ((_p[3]>>0)&0xf);
  }
public:
  uint32_t _p[4];
};

class Validator {
public:
  Validator(const Configuration& cfg);
public:
  void validate(const char* buffer, int ret);
public:
  static void set_verbose(bool);
  static void dump_rates ();
  static void dump_totals();
private:
  virtual bool _validate_raw(const XtcData::Transition& transition,
                             const StreamHeader&        s) = 0;
  virtual bool _validate_fex(const StreamHeader&        s) = 0;
protected:
  const Configuration& _cfg;
  XtcData::Transition  _transition;
  unsigned             _neventcnt;
};

class Fmc126Validator : public Validator {
public:
  Fmc126Validator(const Configuration& cfg,
                  unsigned testpattern=0);
private:
  virtual bool _validate_raw(const XtcData::Transition& transition,
                             const StreamHeader&        s);
  virtual bool _validate_fex(const StreamHeader&        s);
private:
  unsigned             _sample_value;
};

class Fmc134Validator : public Validator {
public:
  Fmc134Validator(const Configuration& cfg,
                  unsigned testpattern=0);
private:
  virtual bool _validate_raw(const XtcData::Transition& transition,
                             const StreamHeader&        s);
  virtual bool _validate_fex(const StreamHeader&        s);
private:
  unsigned             _testpattern;
  unsigned             _sample_value;
  unsigned             _num_samples_raw;
};
