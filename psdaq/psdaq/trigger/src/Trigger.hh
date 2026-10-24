/**
 * @file
 * @brief Trigger, the interface of trigger decision plugins run by the TEB.
 */
#ifndef Pds_Trg_Trigger_hh
#define Pds_Trg_Trigger_hh

#include "psdaq/service/Dl.hh"
#include "psdaq/eb/src/eb.hh"               // For MAX_DRPS
#include "psdaq/eb/src/ResultDgram.hh"
#include "xtcdata/xtc/Dgram.hh"

#include <nlohmann/json.hpp>
#include <cstdint>
#include <string>

namespace Pds {
  namespace Trg {

    /** Interface of the trigger decision code loaded by the TEB: event() fills the result datagram of each built event from its contributions. */
    class Trigger
    {
    public:
      /** Does nothing; empty virtual destructor. */
      virtual ~Trigger() {}
    public:
      /** Return the number of monitor buffers of MEB meb (out of nBufs) to reserve for events containing readout group rog; the TEB sums this over the non-common groups. Returns 0 unless overridden. */
      virtual unsigned rogReserve(unsigned rog,
                                  unsigned meb,
                                  size_t   nBufs) const { return 0; }
      /** Pure virtual: configure from the connect and configure messages and the TEB parameters; implementations return 0 on success. */
      virtual int      configure(const nlohmann::json&      connectMsg,
                                 const nlohmann::json&      configureMsg,
                                 const Pds::Eb::EbParams&   prms) = 0;
      /** Hook given the input region sizes and the result region size after configuration; returns 0 unless overridden. */
      virtual int      initialize(const std::vector<size_t>& inputsRegSizes,
                                  size_t                     resultsRegSize) { return 0; };
      /** Pure virtual: set the trigger decision in result from the contributions in [start, end). The TEB calls it for events and transitions. */
      virtual void     event(const Pds::EbDgram* const* start,
                             const Pds::EbDgram**       end,
                             Pds::Eb::ResultDgram&      result) = 0;
      /** Hook for transitions; does nothing unless overridden. */
      virtual void     transition(Pds::Eb::ResultDgram& result) {}
      /** Hook for shutdown; does nothing unless overridden. */
      virtual void     shutdown() {};
    public:
      /** Return the size of a Pds::Eb::ResultDgram, the result size used by the TEB. */
      static size_t size() { return sizeof(Pds::Eb::ResultDgram); }
    };
  };
};

#endif
