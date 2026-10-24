/**
 * @file
 * @brief RunInfoDef and ChunkInfoDef, the XTC name lists of the run information and chunk information records.
 */
#ifndef Drp_RunInfoDef_hh
#define Drp_RunInfoDef_hh

#include "xtcdata/xtc/NamesId.hh"
#include "xtcdata/xtc/VarDef.hh"

namespace Drp
{

/** VarDef of the runinfo record written by DrpBase::runInfoData(): experiment name and run number. */
class RunInfoDef : public XtcData::VarDef
{
public:
  /** Positions of the RunInfoDef fields. */
  enum index
    {
        EXPT,  ///< Index 0: expt (CHARSTR).
        RUNNUM  ///< Index 1: runnum (UINT32).
    };

  /** Add expt (CHARSTR, rank 1) and runnum (UINT32) to NameVec. */
  RunInfoDef()
   {
       XtcData::VarDef::NameVec.push_back({"expt", XtcData::Name::CHARSTR,1});
       XtcData::VarDef::NameVec.push_back({"runnum", XtcData::Name::UINT32});
   }
};

/** VarDef of the chunkinfo record written by DrpBase::chunkInfoData(): file name and chunk ID. */
class ChunkInfoDef : public XtcData::VarDef
{
public:
  /** Positions of the ChunkInfoDef fields. */
  enum index
    {
        FILENAME,  ///< Index 0: filename (CHARSTR).
        CHUNKID  ///< Index 1: chunkid (UINT32).
    };

  /** Add filename (CHARSTR, rank 1) and chunkid (UINT32) to NameVec. */
  ChunkInfoDef()
   {
       XtcData::VarDef::NameVec.push_back({"filename", XtcData::Name::CHARSTR,1});
       XtcData::VarDef::NameVec.push_back({"chunkid", XtcData::Name::UINT32});
   }
};

}
#endif
