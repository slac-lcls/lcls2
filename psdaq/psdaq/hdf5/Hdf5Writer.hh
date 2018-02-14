
#ifndef HDF5WRITER__H
#define HDF5WRITER__H

#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/TypeId.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/XtcIterator.hh"
#include "xtcdata/xtc/Dgram.hh"
#include <hdf5.h>
#include <unordered_map>

class Dataset
{
public:
  Dataset(hid_t file_id, XtcData::Name& name, const uint32_t* shape);
    void append(const void* data);
    ~Dataset();
    Dataset(Dataset&& d);
    Dataset(const Dataset&) = delete;
    void operator=(const Dataset&) = delete;

private:
    hid_t m_dsetId, m_dataspaceId, m_plistId, m_typeId;
    std::vector<hsize_t> m_dims;
    std::vector<hsize_t> m_offset;
    std::vector<hsize_t> m_dims_extend;
};

class HDF5File : private XtcData::XtcIterator
{
public:
  HDF5File(const char* name, std::vector<XtcData::NameIndex>& namesVec);
    ~HDF5File();
    void save(XtcData::Dgram& dgram);

private:
    enum { Stop, Continue };
  void addDatasets(XtcData::DescData& desc);
  void appendData(XtcData::DescData& data);
    int process(XtcData::Xtc* xtc)
    {
        switch (xtc->contains.id()) {
        case (XtcData::TypeId::Parent): {
            iterate(xtc);
            break;
        }
        case (XtcData::TypeId::ShapesData): {
	  XtcData::ShapesData& shapesdata = *(XtcData::ShapesData*)xtc;
            // lookup the index of the names we are supposed to use
            unsigned namesId = shapesdata.shapes().namesId();
	    XtcData::DescData descdata(shapesdata, _namesVec[namesId]);

            addDatasets(descdata);
            appendData(descdata);

            break;
        }
        default:
            break;
        }
        return Continue;
    }

    hid_t fileId, faplId;
    std::unordered_map<std::string, Dataset> m_datasets;
  std::vector<XtcData::NameIndex>& _namesVec;
};

#endif // HDF5WRITER__H
