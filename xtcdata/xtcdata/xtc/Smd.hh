/**
 * @file
 * @brief Declares XtcData::Smd, which builds small-data (smd) datagrams holding an offset and a size per L1Accept.
 */
#ifndef XtcData_Smd_hh
#define XtcData_Smd_hh

#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/DescData.hh"

namespace XtcData
{

/** Builds smd datagrams: non-L1Accept transitions are copied (EndRun without payload), and each L1Accept gets a ShapesData with two UINT64 fields, intOffset and intDgramSize. */
class Smd
{
public:

    /** Default constructor; does nothing. */
    Smd() {
    };

    /**
     * Write into buf an smd version of dgIn and return it as a Dgram pointer.
     * For EndRun only the header is copied (empty payload); other non-L1Accept transitions are copied whole, and for Configure a Names xtc (detName "smdinfo", alg "offsetAlg", detType "offset", UINT64 fields intOffset and intDgramSize) is added under namesId and recorded in namesLookup, after throwing a const char* if a Names xtc in the copy already uses namesId.
     * For L1Accept the time and env are copied and the payload is replaced by a ShapesData holding offset and size; a message is printed if size is 0.
     */
    Dgram* generate(Dgram* dgIn, void* buf, const void* bufEnd, uint64_t offset, uint64_t size,
                    NamesLookup& namesLookup, NamesId namesId);

}; // end class Smd

}; // end namespace XtcData


#endif
