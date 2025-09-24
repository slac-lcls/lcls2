#include "FileWriterBase.hh"

void SmdWriter::addNames(XtcData::Xtc& parent, const void* bufEnd, unsigned nodeId)
{
    XtcData::Alg alg("offsetAlg", 0, 0, 0);
    XtcData::NamesId namesId(nodeId, 0);
    XtcData::Names& offsetNames = *new(parent, bufEnd) XtcData::Names(bufEnd, "info", alg, "offset", "", namesId);
    SmdDef smdDef;
    offsetNames.add(parent, bufEnd, smdDef);
    namesLookup[namesId] = XtcData::NameIndex(offsetNames);
}
