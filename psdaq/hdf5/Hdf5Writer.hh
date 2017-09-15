#ifndef HDF5WRITER__H
#define HDF5WRITER__H

#include "xtcdata/xtc/Descriptor.hh"
#include "xtcdata/xtc/TypeId.hh"
#include "xtcdata/xtc/Xtc.hh"
#include "xtcdata/xtc/XtcIterator.hh"
#include <hdf5.h>
#include <unordered_map>

class Dataset
{
public:
    Dataset(hid_t file_id, const Field& field);
    void append(const void* data);
    ~Dataset();
    Dataset(Dataset&& d);
    Dataset(const Dataset&) = delete;
    void operator=(const Dataset&) = delete;

private:
    hid_t m_dsetId, m_dataspaceId, m_plistId, m_typeId;
};

class HDF5File
{
public:
    HDF5File(const char* name);
    void addDatasets(Descriptor& desc);
    void appendData(DescData& data);
    ~HDF5File();

private:
    hid_t fileId, faplId;
    std::unordered_map<std::string, Dataset> m_datasets;
};

class HDF5LevelIter : public XtcIterator
{
public:
    enum { Stop, Continue };
    HDF5LevelIter(XtcData::Xtc* xtc, HDF5File& file) : XtcIterator(xtc), _file(file)
    {
    }

    int process(XtcData::Xtc* xtc)
    {
        switch (xtc->contains.id()) {
        case (TypeId::Parent): {
            iterate(xtc);
            break;
        }
        case (TypeId::DescData): {
            DescData& descdata = *(DescData*)xtc->payload();
            Descriptor& desc = descdata.desc();

            _file.addDatasets(desc);
            _file.appendData(descdata);

            break;
        }
        default:
            printf("TypeId %s (value = %d)\n", TypeId::name(xtc->contains.id()), (int)xtc->contains.id());
            break;
        }
        return Continue;
    }

private:
    HDF5File& _file;
};

#endif // HDF5WRITER__H
