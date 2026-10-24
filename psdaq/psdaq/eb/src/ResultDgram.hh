/**
 * @file
 * @brief ResultDgram, the trigger result datagram (an EbDgram plus a packed result word and a monitor buffer number word), and the ResultType codes.
 */
#ifndef Pds_Eb_ResultDgram_hh
#define Pds_Eb_ResultDgram_hh

#include "eb.hh"

#include "psdaq/service/EbDgram.hh"

#include <cstdint>

/** Store VAL in field FLD of the auxdata() value, using the enclosing class constants v_FLD, m_FLD and s_FLD. Used by CubeResultDgram and WindowResultDgram; WindowResultDgram.hh undefines it. */
#define ADDBITS(FLD,VAL) auxdata((auxdata()&~s_##FLD) | ((VAL&m_##FLD)<<v_##FLD))
/** Extract field FLD from the auxdata() value, using the enclosing class constants s_FLD and v_FLD (the expansion has no outer parentheses). */
#define GETBITS(FLD)     (auxdata()&s_##FLD)>>v_##FLD

namespace Pds {
  namespace Eb {

    // ConfigDgram specializations
    /** Result type codes, described in the original comment as ConfigDgram specializations; CubeConfigDgram::resultType(char*) maps the strings Cube and Window to these codes. */
    enum ResultType { Base=0, /**< Default type (0); also used for unrecognized strings. */ Cube=1, /**< Cube result type (1). */ Window=2  /**< Window result type (2). */ };

    /** Trigger result datagram: an EbDgram followed by a 32-bit result word and a 32-bit monitor buffer number word. The result word holds the prescale bit (bit 0), the persist bit (bit 1), a MAX_MRQS-bit monitor field starting at bit 2 and a 24-bit auxdata field (bits 31-8). */
    class ResultDgram : public Pds::EbDgram
    {
      /* bit field access enums
       *       v is the index of the rightmost bit
       *       k is the number bits in the field
       *       m is the mask, right justified
       *       s is the mask shifted into place
       */
      static constexpr uint32_t v_prescale{0}, k_prescale{1};
      static constexpr uint32_t v_persist {1}, k_persist {1};
      static constexpr uint32_t v_monitor {2}, k_monitor {MAX_MRQS};
      static constexpr uint32_t v_auxdata {8}, k_auxdata {24};

      // monitor is sized by MAX_MRQS, so raising it silently walks into auxdata
      static_assert(v_monitor + k_monitor <= v_auxdata,
                    "monitor field overlaps auxdata: MAX_MRQS is too large");

      static constexpr uint32_t m_prescale{(1 << k_prescale) - 1}, s_prescale{m_prescale << v_prescale};
      static constexpr uint32_t m_persist {(1 << k_persist)  - 1}, s_persist {m_persist  << v_persist };
      static constexpr uint32_t m_monitor {(1 << k_monitor)  - 1}, s_monitor {m_monitor  << v_monitor };
      static constexpr uint32_t m_auxdata {(1 << k_auxdata)  - 1}, s_auxdata {m_auxdata  << v_auxdata };

    public:
      /** Copy the pulse ID and transition header of dgram, use a new Xtc of TypeId::Data (version 0) with source Src(id, Level::Event), zero both words, and add their size to the Xtc extent. */
      ResultDgram(const Pds::EbDgram& dgram, unsigned id) :
        Pds::EbDgram(dgram, XtcData::Dgram(dgram, XtcData::Xtc(XtcData::TypeId(XtcData::TypeId::Data, 0),
                                                               XtcData::Src(id, XtcData::Level::Event)))),
        _data    (0),
        _monBufNo(0)
      {
        xtc.extent += sizeof(ResultDgram) - sizeof(Pds::EbDgram);
      }
    public:
      /** Set or clear the prescale bit (bit 0) of the result word. */
      void     prescale(bool     value) { _data = ((_data & ~s_prescale) |
                                                   (value << v_prescale)); }
      /** Set or clear the persist bit (bit 1) of the result word. */
      void     persist (bool     value) { _data = ((_data & ~s_persist)  |
                                                   (value << v_persist));  }
      /** Set the monitor field (MAX_MRQS bits starting at bit 2) of the result word to value, masked to the field width. */
      void     monitor (uint32_t value) { _data = ((_data & ~s_monitor)  |
                                                   ((value << v_monitor) & s_monitor)); }
      /** Set the auxdata field (bits 31-8) of the result word to value, masked to 24 bits. */
      void     auxdata (uint32_t value) { _data = ((_data & ~s_auxdata)  |
                                                   ((value << v_auxdata) & s_auxdata)); }
      /** Return true if the prescale bit (bit 0) is set. */
      bool     prescale() const { return  _data & s_prescale;              }
      /** Return true if the persist bit (bit 1) is set. */
      bool     persist () const { return  _data & s_persist;               }
      /** Return the monitor field (MAX_MRQS bits starting at bit 2), shifted down. */
      uint32_t monitor () const { return (_data & s_monitor) >> v_monitor; }
      /** Return the auxdata field (bits 31-8), shifted down. */
      uint32_t auxdata () const { return (_data & s_auxdata) >> v_auxdata; }
      /** Return the whole 32-bit result word. */
      uint32_t data()     const { return  _data; }
      /** Set the monitor buffer number word (CubeConfigDgram stores its ResultType in this word). */
      void     monBufNo(uint32_t monBufNo_) { _monBufNo = monBufNo_; }
      /** Return the monitor buffer number word. */
      uint32_t monBufNo() const { return _monBufNo; }
    private:
      uint32_t _data;
      uint32_t _monBufNo;
    };
  };
};

#endif
