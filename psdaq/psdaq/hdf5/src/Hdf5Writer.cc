#include "psdaq/hdf5/Hdf5Writer.hh"
#include <cassert>
#include <hdf5_hl.h>
#include <vector>
#include <iostream>

#define _unused(x) ((void)(x))

Dataset::Dataset(hid_t file_id, XtcData::Name& name, const uint32_t* shape)
{
    int ndims = name.rank() + 1;
    std::vector<hsize_t> maxDims(ndims);
    m_dims.push_back(0);
    m_offset.push_back(0);
    m_dims_extend.push_back(1);
    maxDims[0] = H5S_UNLIMITED;
    for (int i = 1; i < ndims; i++) {
        m_dims.push_back(shape[i - 1]);
        m_offset.push_back(0);
        m_dims_extend.push_back(shape[i - 1]);
        maxDims[i] = shape[i - 1];
    }
    m_dataspaceId = H5Screate_simple(ndims, m_dims.data(), maxDims.data());

    std::vector<hsize_t> cdims(ndims);

    // chunk size for scalars
    if (ndims == 1) {
        cdims[0] = 10000;
    }
    // chunk size for arrays
    else {
        cdims[0] = 1;
        for (int i = 1; i < ndims; i++) {
            cdims[i] = shape[i - 1];
        }
    }

    m_plistId = H5Pcreate(H5P_DATASET_CREATE);
    if (H5Pset_chunk(m_plistId, ndims, cdims.data()) < 0) {
        std::cout << "Error in setting chunk size" << std::endl;
    }

    switch (name.type()) {
    case XtcData::Name::UINT8: {
        m_typeId = H5T_NATIVE_UINT8;
        break;
    }
    case XtcData::Name::UINT16: {
        m_typeId = H5T_NATIVE_UINT16;
        break;
    }
    case XtcData::Name::INT32: {
        m_typeId = H5T_NATIVE_INT32;
        break;
    }
    case XtcData::Name::FLOAT: {
        m_typeId = H5T_NATIVE_FLOAT;
        break;
    }
    case XtcData::Name::DOUBLE: {
        m_typeId = H5T_NATIVE_DOUBLE;
        break;
    }
    }
    m_dsetId = H5Dcreate2(file_id, name.name(), m_typeId, m_dataspaceId, H5P_DEFAULT, m_plistId, H5P_DEFAULT);
    if (m_dsetId < 0) {
        std::cout << "Error in creating HDF5 dataset " << std::string(name.name()) << std::endl;
    }
}

void Dataset::append(const void* data)
{
    m_dims[0]+=1;
    herr_t status = H5Dset_extent (m_dsetId, m_dims.data());
    _unused(status);
    hid_t filespace = H5Dget_space (m_dsetId);
    status = H5Sselect_hyperslab (filespace, H5S_SELECT_SET, m_offset.data(), NULL,
                                  m_dims_extend.data(), NULL);  

    /* Define memory space */
    hid_t memspace = H5Screate_simple (m_dims.size(), m_dims_extend.data(), NULL); 

    /* Write the data to the extended portion of dataset  */
    status = H5Dwrite (m_dsetId, m_typeId, memspace, filespace,
                       H5P_DEFAULT, data);
    m_offset[0]+=1;

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

    m_dims = std::move(d.m_dims);
    m_offset = std::move(d.m_offset);
    m_dims_extend = std::move(d.m_dims_extend);
}

HDF5File::HDF5File(const char* name, std::vector<XtcData::NameIndex>& namesVec) :
    XtcIterator(), _namesVec(namesVec)
{
    faplId = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(faplId, H5F_LIBVER_EARLIEST, H5F_LIBVER_LATEST);
    fileId = H5Fcreate(name, H5F_ACC_TRUNC, H5P_DEFAULT, faplId);
    if (fileId < 0) {
        std::cout << "Could not create HDF5 file:  " << std::endl;
    }
}

void HDF5File::save(XtcData::Dgram& dgram) {
    iterate(&dgram.xtc);
}

void HDF5File::addDatasets(XtcData::DescData& descdata)
{
    XtcData::Names& names = descdata.nameindex().names();
    for (unsigned i = 0; i < names.num(); i++) {
        XtcData::Name& name = names.get(i);
        const char* namechar = name.name();
        std::string namestr(namechar);
        auto it = m_datasets.find(namestr);
        if (it == m_datasets.end()) {
            // only create dataset if it doesn't exist
            uint32_t* shape = 0;
            if (name.rank()>0) shape = descdata.shape(name);
            m_datasets.emplace(namestr, Dataset(fileId, name, shape));
        }
    }
}

void HDF5File::appendData(XtcData::DescData& descdata)
{
    XtcData::Names& names = descdata.nameindex().names();
    for (unsigned i = 0; i < names.num(); i++) {
        XtcData::Name& name = names.get(i);
        const char* namechar = name.name();
        std::string namestr(namechar);
        auto it = m_datasets.find(namestr);
        assert(it != m_datasets.end());
        unsigned index = descdata.nameindex().nameMap()[namechar];
        it->second.append(descdata.address(index));
    }
}

HDF5File::~HDF5File()
{
    H5Fclose(fileId);
    H5Pclose(faplId);
}
