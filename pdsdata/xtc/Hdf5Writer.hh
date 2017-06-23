#ifndef HDF5WRITER__H
#define HDF5WRITER__H

#include <unordered_map>
#include <hdf5.h>
#include "pdsdata/xtc/Descriptor.hh"


class Dataset
{
public:
    Dataset(hid_t file_id, const Field& field);
    void append(const void* data);
    ~Dataset();
    Dataset(Dataset&& d);
    Dataset(const Dataset&) = delete;
    void operator = (const Dataset&) = delete;
private:
    hid_t m_dsetId, m_dataspaceId, m_plistId, m_typeId;
};

class HDF5File
{
public:
    HDF5File(const char* name);
    void addDatasets(Descriptor& desc);
    void appendData(Data& data);
    ~HDF5File();
private:
    hid_t fileId, faplId;
    std::unordered_map<std::string, Dataset> m_datasets;
};

#endif // HDF5WRITER__H
