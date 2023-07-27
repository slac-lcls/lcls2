#include "BldNames.hh"
#include "xtcdata/xtc/VarDef.hh"
#include "xtcdata/xtc/ShapesData.hh"

using namespace XtcData;

BldNames::EBeamDataV7::EBeamDataV7() {
    NameVec.push_back(Name("damageMask"       , Name::UINT32));
    NameVec.push_back(Name("ebeamCharge"      , Name::DOUBLE));
    NameVec.push_back(Name("ebeamL3Energy"    , Name::DOUBLE));
    NameVec.push_back(Name("ebeamLTUPosX"     , Name::DOUBLE));
    NameVec.push_back(Name("ebeamLTUPosY"     , Name::DOUBLE));
    NameVec.push_back(Name("ebeamLUTAngX"     , Name::DOUBLE));
    NameVec.push_back(Name("ebeamLTUAngY"     , Name::DOUBLE));
    NameVec.push_back(Name("ebeamPkCurrBC2"   , Name::DOUBLE));
    NameVec.push_back(Name("ebeamEnergyBC2"   , Name::DOUBLE));
    NameVec.push_back(Name("ebeamPkCurrBC1"   , Name::DOUBLE));
    NameVec.push_back(Name("ebeamEnergyBC1"   , Name::DOUBLE));
    NameVec.push_back(Name("ebeamUndPosX"     , Name::DOUBLE));
    NameVec.push_back(Name("ebeamUndPosY"     , Name::DOUBLE));
    NameVec.push_back(Name("ebeamUndAngX"     , Name::DOUBLE));
    NameVec.push_back(Name("ebeamUndAngY"     , Name::DOUBLE));
    NameVec.push_back(Name("ebeamXTCAVAmpl"   , Name::DOUBLE));
    NameVec.push_back(Name("ebeamXTCAVPhase"  , Name::DOUBLE));
    NameVec.push_back(Name("ebeamDumpCharge"  , Name::DOUBLE));
    NameVec.push_back(Name("ebeamPhotonEnergy", Name::DOUBLE));
    NameVec.push_back(Name("ebeamLTU250"      , Name::DOUBLE));
    NameVec.push_back(Name("ebeamLTU450"      , Name::DOUBLE));
}

BldNames::PCav::PCav() {
    NameVec.push_back(Name("fitTime1"      , Name::DOUBLE));
    NameVec.push_back(Name("fitTime2"      , Name::DOUBLE));
    NameVec.push_back(Name("charge1"       , Name::DOUBLE));
    NameVec.push_back(Name("charge2"       , Name::DOUBLE));
}

BldNames::GmdV1::GmdV1() {
    NameVec.push_back(Name("energy"      , Name::DOUBLE));
    NameVec.push_back(Name("xpos"        , Name::DOUBLE));
    NameVec.push_back(Name("ypos"        , Name::DOUBLE));
    NameVec.push_back(Name("avgIntensity", Name::DOUBLE));
    NameVec.push_back(Name("rmsElectronSum", Name::INT64));
    NameVec.push_back(Name("electron1BkgNoiseAvg", Name::INT16));
    NameVec.push_back(Name("electron2BkgNoiseAvg", Name::INT16));
}

BldNames::IpimbDataV2::IpimbDataV2() {
    NameVec.push_back(Name("triggerCounter", Name::UINT64));
    NameVec.push_back(Name("config"        , Name::UINT16, 1));
    NameVec.push_back(Name("channel"       , Name::UINT16, 1));
    NameVec.push_back(Name("channelps"     , Name::UINT16, 1));
    NameVec.push_back(Name("checksum"      , Name::UINT16));
}

BldNames::IpimbConfigV2::IpimbConfigV2() {
    NameVec.push_back(Name("triggerCounter"     , Name::UINT64));
    NameVec.push_back(Name("serialID"           , Name::UINT64));
    NameVec.push_back(Name("chargeAmpRange"     , Name::UINT16));
    NameVec.push_back(Name("calibrationRange"   , Name::UINT16));
    NameVec.push_back(Name("resetLength"        , Name::UINT32));
    NameVec.push_back(Name("resetDelay"         , Name::UINT32));
    NameVec.push_back(Name("chargeAmpRefVoltage", Name::FLOAT));
    NameVec.push_back(Name("calibrationVoltage" , Name::FLOAT));
    NameVec.push_back(Name("diodeBias"          , Name::FLOAT));
    NameVec.push_back(Name("status"             , Name::UINT16));
    NameVec.push_back(Name("errors"             , Name::UINT16));
    NameVec.push_back(Name("calStrobeLength"    , Name::UINT16));
    NameVec.push_back(Name("trigDelay"          , Name::UINT32));
    NameVec.push_back(Name("trigPsDelay"        , Name::UINT32));
    NameVec.push_back(Name("adcDelay"           , Name::UINT32));
}

BldNames::GmdV2::GmdV2() { NameVec.push_back(Name("millijoulesperpulse",Name::FLOAT,0)); }

BldNames::XGmdV2::XGmdV2() { 
    NameVec.push_back(Name("millijoulesperpulse",Name::FLOAT,0));
    NameVec.push_back(Name("POSY",Name::FLOAT,0)); 
}

