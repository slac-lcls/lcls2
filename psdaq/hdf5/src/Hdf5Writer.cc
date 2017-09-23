#include "psdaq/hdf5/Hdf5Writer.hh"
#include <cassert>
#include <hdf5_hl.h>
#include <vector>

Dataset::Dataset(hid_t file_id, const Field& field)
{
    int ndims = field.rank + 1;
    std::vector<hsize_t> dims(ndims);
    std::vector<hsize_t> maxDims(ndims);
    dims[0] = 0;
    maxDims[0] = H5S_UNLIMITED;
    for (int i = 1; i < ndims; i++) {
        dims[i] = field.shape[i - 1];
        maxDims[i] = field.shape[i - 1];
    }
    m_dataspaceId = H5Screate_simple(ndims, dims.data(), maxDims.data());

    std::vector<hsize_t> cdims(ndims);

    // chunk size for scalars
    if (ndims == 1) {
        cdims[0] = 10000;
    }
    // chunk size for arrays
    else {
        cdims[0] = 1;
        for (int i = 1; i < ndims; i++) {
            cdims[i] = field.shape[i - 1];
        }
    }

    m_plistId = H5Pcreate(H5P_DATASET_CREATE);
    if (H5Pset_chunk(m_plistId, ndims, cdims.data()) < 0) {
        std::cout << "Error in setting chunk size" << std::endl;
    }

    switch (field.type) {
    case UINT8: {
        m_typeId = H5T_NATIVE_UINT8;
        break;
    }
    case UINT16: {
        m_typeId = H5T_NATIVE_UINT16;
        break;
    }
    case INT32: {
        m_typeId = H5T_NATIVE_INT32;
        break;
    }
    case FLOAT: {
        m_typeId = H5T_NATIVE_FLOAT;
        break;
    }
    case DOUBLE: {
        m_typeId = H5T_NATIVE_DOUBLE;
        break;
    }
    }
    m_dsetId = H5Dcreate2(file_id, field.name, m_typeId, m_dataspaceId, H5P_DEFAULT, m_plistId, H5P_DEFAULT);
    if (m_dsetId < 0) {
        std::cout << "Error in creating HDF5 dataset " << field.name << std::endl;
    }
}

void Dataset::append(const void* data)
{
    printf("append not supported with hdf5 1.8\n");
    // H5DOappend(m_dsetId, H5P_DEFAULT, 0, 1, m_typeId, data);
    // H5Dflush(m_dsetId);
}

Dataset::~Dataset()
{
    if (m_dsetId >= 0) {
        H5Dclose(m_dsetId);
    }
    if (m_plistId >= 0) {
        H5Pclose(m_plistId);
    }
    if (m_dataspaceId >= 0) {
        H5Sclose(m_dataspaceId);
    }
}

Dataset::Dataset(Dataset&& d)
{
    m_dsetId = d.m_dsetId;
    m_dataspaceId = d.m_dataspaceId;
    m_plistId = d.m_plistId;
    m_typeId = d.m_typeId;

    d.m_dsetId = -1;
    d.m_dataspaceId = -1;
    d.m_plistId = -1;
    d.m_typeId = -1;
}

HDF5File::HDF5File(const char* name)
{
    faplId = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(faplId, H5F_LIBVER_EARLIEST, H5F_LIBVER_LATEST);
    fileId = H5Fcreate(name, H5F_ACC_TRUNC, H5P_DEFAULT, faplId);
    if (fileId < 0) {
        std::cout << "Could not create HDF5 file:  " << std::endl;
    }
}

void HDF5File::addDatasets(Desc& desc)
{
    for (unsigned i = 0; i < desc.num_fields(); i++) {
        Field& field = desc.get(i);
        auto it = m_datasets.find(field.name);
        if (it == m_datasets.end()) {
            // only create dataset if it doesn't exist
            m_datasets.emplace(field.name, Dataset(fileId, field));
        }
    }
}

void HDF5File::appendData(DescData& datadesc)
{
    Desc& desc = datadesc.desc();
    char* data = datadesc.data().payload();
    for (unsigned i = 0; i < desc.num_fields(); i++) {
        Field& field = desc.get(i);
        auto it = m_datasets.find(field.name);
        assert(it != m_datasets.end());
        it->second.append(data + field.offset);
    }
}

HDF5File::~HDF5File()
{
    H5Fclose(fileId);
    H5Pclose(faplId);
}
