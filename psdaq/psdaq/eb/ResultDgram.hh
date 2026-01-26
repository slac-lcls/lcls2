#ifndef Pds_Eb_ResultDgram_hh
#define Pds_Eb_ResultDgram_hh

#include "eb.hh"

#include "psdaq/service/EbDgram.hh"

#include <cstdint>


namespace Pds {
  namespace Eb {

    class ResultDgram : public Pds::EbDgram
    {
      /* bit field access enums
       *       v is the index of the rightmost bit
       *       k is the number bits in the field
       *       m is the mask, right justified
       *       s is the mask shifted into place
       */
      enum { v_prescale = 0, k_prescale = 1 };
      enum { v_persist  = 1, k_persist  = 1 };
      enum { v_monitor  = 2, k_monitor  = MAX_MRQS };
      enum { v_auxdata  = 8, k_auxdata  = 24 };

      enum { m_prescale = ((1 << k_prescale) - 1), s_prescale = (m_prescale << v_prescale) };
      enum { m_persist  = ((1 << k_persist)  - 1), s_persist  = (m_persist  << v_persist)  };
      enum { m_monitor  = ((1 << k_monitor)  - 1), s_monitor  = (m_monitor  << v_monitor)  };
      enum { m_auxdata  = ((1 << k_auxdata)  - 1), s_auxdata  = (m_auxdata  << v_auxdata)  };

    public:
      ResultDgram(const Pds::EbDgram& dgram, unsigned id) :
        Pds::EbDgram(dgram, XtcData::Dgram(dgram, XtcData::Xtc(XtcData::TypeId(XtcData::TypeId::Data, 0),
                                                               XtcData::Src(id, XtcData::Level::Event)))),
        _data(0),
        _monBufNo(0)
      {
        xtc.extent += sizeof(ResultDgram) - sizeof(Pds::EbDgram);
      }
    public:
      void     prescale(bool     value) { _data = ((_data & ~s_prescale) |
                                                   (value << v_prescale)); }
      void     persist (bool     value) { _data = ((_data & ~s_persist)  |
                                                   (value << v_persist));  }
      void     monitor (uint32_t value) { _data = ((_data & ~s_monitor)  |
                                                   ((value << v_monitor) & s_monitor)); }
      void     auxdata (uint32_t value) { _data = ((_data & ~s_auxdata)  |
                                                   ((value << v_auxdata) & s_auxdata)); }
      bool     prescale() const { return  _data & s_prescale;              }
      bool     persist () const { return  _data & s_persist;               }
      uint32_t monitor () const { return (_data & s_monitor) >> v_monitor; }
      uint32_t auxdata () const { return (_data & s_auxdata) >> v_auxdata; }
      uint32_t data()     const { return  _data; }
      void     monBufNo(uint32_t monBufNo_) { _monBufNo = monBufNo_; }
      uint32_t monBufNo() const { return _monBufNo; }
    private:
      uint32_t _data;
      uint32_t _monBufNo;
    };
  };
};

#endif
