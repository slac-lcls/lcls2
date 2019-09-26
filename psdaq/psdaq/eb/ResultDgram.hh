#ifndef Pds_Eb_ResultDgram_hh
#define Pds_Eb_ResultDgram_hh

#include "xtcdata/xtc/Dgram.hh"

#include <cstdint>


namespace Pds {
  namespace Eb {

    class ResultDgram : public XtcData::Dgram
    {
      /* bit field access enums
       *       v is the index of the rightmost bit
       *       k is the number bits in the field
       *       m is the mask, right justified
       *       s is the mask shifted into place
       */
      enum { v_persist  = 0, k_persist  = 1 };
      enum { v_monitor  = 1, k_monitor  = 1 };
      enum { v_prescale = 2, k_prescale = 1 };

      enum { m_persist  = ((1 << k_persist)  - 1), s_persist  = (m_persist  << v_persist)  };
      enum { m_monitor  = ((1 << k_monitor)  - 1), s_monitor  = (m_monitor  << v_monitor)  };
      enum { m_prescale = ((1 << k_prescale) - 1), s_prescale = (m_prescale << v_prescale) };

      enum { m_persists  =
             ((s_persist  << 0*4) | (s_persist  << 1*4) | (s_persist  << 2*4) | (s_persist  << 3*4) |
              (s_persist  << 4*4) | (s_persist  << 5*4) | (s_persist  << 6*4) | (s_persist  << 7*4)) };
      enum { m_monitors  =
             ((s_monitor  << 0*4) | (s_monitor  << 1*4) | (s_monitor  << 2*4) | (s_monitor  << 3*4) |
              (s_monitor  << 4*4) | (s_monitor  << 5*4) | (s_monitor  << 6*4) | (s_monitor  << 7*4)) };
      enum { m_prescales =
             ((s_prescale << 0*4) | (s_prescale << 1*4) | (s_prescale << 2*4) | (s_prescale << 3*4) |
              (s_prescale << 4*4) | (s_prescale << 5*4) | (s_prescale << 6*4) | (s_prescale << 7*4)) };
    public:
      ResultDgram(const XtcData::Transition& transition_, unsigned id) :
        XtcData::Dgram(transition_, XtcData::Xtc(XtcData::TypeId(XtcData::TypeId::Data, 0),
                                                 XtcData::Src(id, XtcData::Level::Event))),
        _data(0),
        _monBufNo(0)
      {
        xtc.alloc(sizeof(ResultDgram) - sizeof(XtcData::Dgram));
      }
    public:
      void     persist (unsigned line, bool value) { if (value) _data |=   s_persist  << (4 * line);
                                                     else       _data &= ~(s_persist  << (4 * line)); }
      void     monitor (unsigned line, bool value) { if (value) _data |=   s_monitor  << (4 * line);
                                                     else       _data &= ~(s_monitor  << (4 * line)); }
      void     prescale(unsigned line, bool value) { if (value) _data |=   s_prescale << (4 * line);
                                                     else       _data &= ~(s_prescale << (4 * line)); }
      bool     persist (unsigned line) const { return (_data >> (4 * line)) & s_persist;  }
      bool     monitor (unsigned line) const { return (_data >> (4 * line)) & s_monitor;  }
      bool     prescale(unsigned line) const { return (_data >> (4 * line)) & s_prescale; }
      uint32_t persist () const { return _data & m_persists;  }
      uint32_t monitor () const { return _data & m_monitors;  }
      uint32_t prescale() const { return _data & m_prescales; }

      void     monBufNo(uint32_t monBufNo_) { _monBufNo = monBufNo_; }
      uint32_t monBufNo() const { return _monBufNo; }
    private:
      uint32_t _data;
      uint32_t _monBufNo;
    };
  };
};

#endif
