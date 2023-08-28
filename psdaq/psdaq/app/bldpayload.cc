//
//  Parse the bld payload PV
//
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <strings.h>
#include <pthread.h>
#include <string>
#include <signal.h>

#include "AppUtils.hh"
#include "psdaq/epicstools/PVBase.hh"
#include "xtcdata/xtc/ShapesData.hh"

#include "psalg/utils/SysLog.hh"
using logging = psalg::SysLog;

static const XtcData::Name::DataType xtype[] = {
    XtcData::Name::UINT8 , // pvBoolean
    XtcData::Name::INT8  , // pvByte
    XtcData::Name::INT16,  // pvShort
    XtcData::Name::INT32 , // pvInt
    XtcData::Name::INT64 , // pvLong
    XtcData::Name::UINT8 , // pvUByte
    XtcData::Name::UINT16, // pvUShort
    XtcData::Name::UINT32, // pvUInt
    XtcData::Name::UINT64, // pvULong
    XtcData::Name::FLOAT , // pvFloat
    XtcData::Name::DOUBLE, // pvDouble
    XtcData::Name::CHARSTR, // pvString
};

class BldDescriptor : public Pds_Epics::PVBase
{
public:
    BldDescriptor(const char* channelName) : Pds_Epics::PVBase("pva",channelName) {}
    ~BldDescriptor();
    XtcData::VarDef get(unsigned& payloadSize);
};

BldDescriptor::~BldDescriptor()
{
    logging::debug("~BldDescriptor");
}

XtcData::VarDef BldDescriptor::get(unsigned& payloadSize)
{
    payloadSize = 0;
    XtcData::VarDef vd;
    const pvd::StructureConstPtr& s = _strct->getStructure();
    if (!s) {
        logging::error("BLD with no payload.  Is FieldMask empty?");
        throw std::string("BLD with no payload.  Is FieldMask empty?");
    }

    const pvd::Structure* structure = static_cast<const pvd::Structure*>(s->getFields()[0].get());

    const pvd::StringArray& names = structure->getFieldNames();
    const pvd::FieldConstPtrArray& fields = structure->getFields();
    logging::debug("BldDescriptor::get found %u/%u fields", names.size(), fields.size());

    vd.NameVec.push_back(XtcData::Name("severity",XtcData::Name::UINT64));
    payloadSize += 8;

    for (unsigned i=0; i<fields.size(); i++) {
        pvd::Type field_t = fields[i]->getType();
        logging::debug("Field %u  type %u",i,field_t);
        switch (field_t) {
            case pvd::scalar: {
                const pvd::Scalar* scalar = static_cast<const pvd::Scalar*>(fields[i].get());
                XtcData::Name::DataType type = xtype[scalar->getScalarType()];
                vd.NameVec.push_back(XtcData::Name(names[i].c_str(), type));
                payloadSize += XtcData::Name::get_element_size(type);
                break;
            }

            default: {
                logging::error("PV type %s for field %s not supported",
                               pvd::TypeFunc::name(field_t),
                               names[i]);
                throw std::string("PV type ")+pvd::TypeFunc::name(fields[i]->getType())+
                                  " for field "+names[i]+" not supported";
                break;
            }
        }
    }

    std::string fnames("fields: ");
    for(auto & elem: vd.NameVec)
        fnames += std::string(elem.name()) + "[" + elem.str_type() + "],";
    logging::debug("%s",fnames.c_str());

    return vd;
}

void usage(const char* p) {
  printf("Usage: %s -n <pvname>\n",p);
}

int main(int argc, char* argv[])
{
  extern char* optarg;
  char c;

  const char* name = 0;
  bool lverbose = false;

  while ( (c=getopt( argc, argv, "n:vh")) != EOF ) {
    switch(c) {
    case 'n':
      name = optarg;
      break;
    case 'v':
      lverbose = true;
      break;
    default:
      usage(argv[0]);
      return 0;
    }
  }

  if (!name) {
    usage(argv[0]);
    return -1;
  }

  logging::init("bldpayload", lverbose ? LOG_DEBUG : LOG_INFO);

  unsigned payloadSz=0;
  BldDescriptor d(name);
  while(!d.ready()) {
      logging::debug("Waiting for connection");
      usleep(1000000);
  }
  d.get(payloadSz);

  printf("payload size is %u\n",payloadSz);
  
  return 1;
}
