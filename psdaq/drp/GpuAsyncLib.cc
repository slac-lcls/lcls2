

#include "GpuAsyncLib.h"

DataGPU::DataGPU(const char* path) {
    fd_ = open(path, O_RDWR);
    if (fd_ < 0) {
        perror("Error while opening file");
        throw "Error while opening file";
    }
}

bool CudaContext::init(int device, bool quiet) {
    CUresult status;
    if ((status = cuInit(0)) != CUDA_SUCCESS) {
        fprintf(stderr, "Error while initting cuda, code %d\n", status);
        return false;
    }

    int devs = 0;
    checkError(cuDeviceGetCount(&devs));
    if (!quiet) fprintf(stderr, "Total devices %d\n", devs);
    if (devs <= 0) {
        fprintf(stderr, "No devices available!\n");
        return false;
    }

    device = device < 0 ? 0 : device;
    if (devs <= device) {
        fprintf(stderr, "Invalid device number %d! There are only %d devices available\n", device, devs);
        return false;
    }

    // Actually get the device...
    if ((status = cuDeviceGet(&device_, device)) != CUDA_SUCCESS) {
        fprintf(stderr, "Could not get device! code=%d\n", status);
        return false;
    }

    // Spew device name
    char name[256];
    checkError(cuDeviceGetName(name, sizeof(name), device_));
    if (!quiet) fprintf(stderr, "Selected device: %s\n", name);

    // Set required attributes
    int res;
    checkError(cuDeviceGetAttribute(&res, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1, device_));
    if (!res) {
        fprintf(stderr, "WARNING: This device does not support CUDA Stream Operations, this code will not run!\n");
        fprintf(stderr,
                "  Consider setting NVreg_EnableStreamMemOPs=1 when loading the NVIDIA kernel module, if your GPU is "
                "supported.\n");
        return false;
    }

    // Report memory totals
    size_t global_mem = 0;
    checkError(cuDeviceTotalMem(&global_mem, device_));
    if (!quiet) fprintf(stderr, "Global memory: %llu MB\n", (unsigned long long)(global_mem >> 20));
    if (!quiet && global_mem > (unsigned long long)4 * 1024 * 1024 * 1024L) fprintf(stderr, "64-bit Memory Address support\n");

    // Create context
    checkError(cuCtxCreate(&context_, 0, device_));

    return true;
}

void CudaContext::list_devices() {
    int devs = 0;
    CUresult status;
    if ((status = cuDeviceGetCount(&devs)) != CUDA_SUCCESS) {
        fprintf(stderr, "Unable to get device count\n");
        return;
    }

    for (int i = 0; i < devs; ++i) {
        CUdevice dev;
        if (cuDeviceGet(&dev, i) != CUDA_SUCCESS) {
            fprintf(stderr, "Unable to get device %d\n", i);
            continue;
        }
        char name[256];
        cuDeviceGetName(name, sizeof(name), dev);
        printf("%d: %s\n", i, name);
    }
}


// -------------------------------------------------------------------

void checkError(CUresult status) {
    if (status != CUDA_SUCCESS) {
        const char* perrstr = 0;
        CUresult ok         = cuGetErrorString(status, &perrstr);
        if (ok == CUDA_SUCCESS) {
            if (perrstr) {
                fprintf(stderr, "status %i: info: %s\n", status, perrstr);
            } else {
                fprintf(stderr, "status %i: info: unknown error\n", status);
            }
        }
        abort();
    }
}

void checkError(cudaError status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "status %i: info: %s\n", status, cudaGetErrorString(status));
        abort();
    }
}

//-----------------------------------------------------------------------------

bool wasError(CUresult status) {
    if (status != CUDA_SUCCESS) {
        const char* perrstr = 0;
        CUresult ok         = cuGetErrorString(status, &perrstr);
        if (ok == CUDA_SUCCESS) {
            if (perrstr) {
                fprintf(stderr, "status %i: info: %s\n", status, perrstr);
            } else {
                fprintf(stderr, "status %i: info: unknown error\n", status);
            }
        }
        return true;
    }
    return false;
}

int gpuInitBufferState(GpuBufferState_t* b, const DataGPU& gpu, size_t bufSize)
{
    // Allocate buffers on the GPU
    if (gpuMapFpgaMem(&b->bwrite, gpu.fd(), 0, bufSize, 1) != 0) {
        perror("gpuMapFpgaMem: write");
        return -1;
    }

    if (gpuMapFpgaMem(&b->bread, gpu.fd(), 0, bufSize, 0) != 0) {
        perror("gpuMapFpgaMem: read");
        gpuUnMapFpgaMem(&b->bwrite);
        return -1;
    }

    return 0;
}

void gpuDestroyBufferState(GpuBufferState_t* b)
{
    gpuUnMapFpgaMem(&b->bwrite);
    gpuUnMapFpgaMem(&b->bread);
}

int gpuMapHostFpgaMem(GpuDmaBuffer_t* outmem, int fd, uint64_t offset, size_t size)
{
    memset(outmem, 0, sizeof(*outmem));

    outmem->ptr = (uint8_t*)dmaMapRegister(fd, offset, size);
    if (!outmem->ptr || outmem->ptr == MAP_FAILED) {
        return -1;
    }
    outmem->size = size;
    outmem->fd = fd;

    CUresult status = cuMemHostRegister(outmem->ptr, size, CU_MEMHOSTREGISTER_IOMEMORY);
    if (wasError(status)) {
        fprintf(stderr, "Unable to map offset=%lu, size=%zu to GPU\n", offset, size);
        dmaUnMapRegister(fd, (void**)outmem->ptr, outmem->size);
        outmem = {};
        return -1;
    }

    status = cuMemHostGetDevicePointer(&outmem->dptr, outmem->ptr, 0);
    if (wasError(status)) {
        fprintf(stderr, "Failed to get device pointer: %d\n", status);
        dmaUnMapRegister(fd, (void**)outmem->ptr, outmem->size);
        outmem = {};
        return -1;
    }

    int flag = 1;
    checkError(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, outmem->dptr));

    return 0;
}

int gpuMapFpgaMem(GpuDmaBuffer_t* outmem, int fd, uint64_t offset, size_t size, int write)
{
    memset(outmem, 0, sizeof(*outmem));

    if (cuMemAlloc(&outmem->dptr, size) != CUDA_SUCCESS) {
        return -1;
    }
    outmem->size = size;
    cuMemsetD8(outmem->dptr, 0, size);

    int flag = 1;
    // This attribute is required for peer shared memory. It will synchronize every synchronous memory operation on this block of memory.
    if (cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, outmem->dptr) != CUDA_SUCCESS) {
        cuMemFree(outmem->dptr);
        outmem = {};
        return -1;
    }

    if (gpuAddNvidiaMemory(fd, write, outmem->dptr, outmem->size) < 0) {
        fprintf(stderr, "gpuAddNvidiaMemory failed\n");
        abort();
        cuMemFree(outmem->dptr);
        outmem = {};
        return -1;
    }

    outmem->size = size;
    outmem->gpuOnly = 1;
    outmem->fd = fd;

    return 0;
}

void gpuUnMapFpgaMem(GpuDmaBuffer_t* mem)
{
    if (!mem->gpuOnly) {
        dmaUnMapRegister(mem->fd, &mem->ptr, mem->size);
        mem->ptr = NULL;
        mem->size = 0;
    }
    // FIXME: gpuOnly memory cannot be unmapped?
}

DmaDest_t dmaDestGet(int fd)
{
    // @todo: This line addresses only lane 0
    uint32_t regVal;
    dmaReadRegister(fd, 0x0002802c, &regVal);

    DmaDest_t mode;
    switch (regVal) {
        case CPU:  mode = CPU;  break;
        case GPU:  mode = GPU;  break;
        default:   mode = ERR;  break;
    }
    return mode;
}

void dmaDestSet(int fd, DmaDest_t mode_)
{
    // @todo: This line addresses only lane 0
    dmaWriteRegister(fd, 0x0002802c, mode_);
}
