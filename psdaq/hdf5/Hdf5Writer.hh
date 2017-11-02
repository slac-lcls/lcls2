#ifndef HDF5WRITER__H
#define HDF5WRITER__H

#include "xtcdata/xtc/DescData.hh"
#include "xtcdata/xtc/TypeId.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/XtcIterator.hh"
#include <hdf5.h>
#include <unordered_map>

class Dataset
{
public:
    Dataset(hid_t file_id, Name& name, const uint32_t* shape);
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

class HDF5File
{
public:
    HDF5File(const char* name);
    void addDatasets(DescData& desc);
    void appendData(DescData& data);
    ~HDF5File();

private:
    hid_t fileId, faplId;
    std::unordered_map<std::string, Dataset> m_datasets;
};

class HDF5Iter : public XtcData::XtcIterator
{
public:
    enum { Stop, Continue };
    HDF5Iter(XtcData::Xtc* xtc, HDF5File& file, std::vector<NameIndex>& namesVec) :
        XtcIterator(xtc), _file(file), _namesVec(namesVec)
    {
    }

    int process(XtcData::Xtc* xtc)
    {
        switch (xtc->contains.id()) {
        case (XtcData::TypeId::Parent): {
            iterate(xtc);
            break;
        }
        case (XtcData::TypeId::ShapesData): {
            ShapesData& shapesdata = *(ShapesData*)xtc;
            // lookup the index of the names we are supposed to use
            unsigned namesId = shapesdata.shapes().namesId();
            DescData descdata(shapesdata, _namesVec[namesId]);

            _file.addDatasets(descdata);
            _file.appendData(descdata);

            break;
        }
        default:
            break;
        }
        return Continue;
    }

private:
    HDF5File& _file;
    std::vector<NameIndex> _namesVec;
};

#endif // HDF5WRITER__H
