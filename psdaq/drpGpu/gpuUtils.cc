#include "gpuUtils.hh"

#include "psdaq/service/Affinity.hh"   // numaNodeOfPciDev
#include "psalg/utils/SysLog.hh"

#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <cstring>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <linux/mempolicy.h>


using logging = psalg::SysLog;
using namespace Drp::Gpu;


// -------------------------------------------------------------------

bool CudaContext::init(int device, bool quiet) {

    int devCount = 0;
    if (chkError(cudaGetDeviceCount(&devCount)))
        return false;
    logging::debug("Total GPU devices: %d", devCount);
    if (devCount <= 0) {
        logging::error("No GPU devices available!");
        return false;
    }

    if (!quiet)  listDevices();

    device = device < 0 ? 0 : device;
    if (devCount <= device) {
        logging::error("Invalid GPU device number %d! There are only %d devices available", device, devCount);
        return false;
    }
    _devNo = device;

    // Actually get the device...
    CUresult status;
    if ((status = cuDeviceGet(&device_, device)) != CUDA_SUCCESS) {
        logging::error("Failed to get GPU device %d: code=%d", device, status);
        return false;
    }

    // Spew device name, compute capability and PCIe bus ID
    cudaDeviceProp prop;
    chkError(cudaGetDeviceProperties(&prop, device_));
    char pciBusId[16];
    chkError(cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), device_));
    logging::info("GPU device %d: %s, CC: %d.%d, Bus-Id: %s",
                  device, prop.name, prop.major, prop.minor, pciBusId);

    // Report memory totals
    size_t global_mem = 0;
    if (chkError(cuDeviceTotalMem(&global_mem, device_)))
        return false;
    logging::info("Global memory: %zu MB", global_mem >> 20);
    if (global_mem > (size_t)4 << 30)
        logging::debug("64-bit Memory Address support");

    // Create context
#if CUDA_VERSION >= 13000
    constexpr unsigned flags = CU_CTX_SCHED_SPIN; // @todo: Revisit
    if (chkError(cuCtxCreate(&context_, NULL, flags, device_)))
#else
    if (chkError(cuCtxCreate(&context_, 0, device_)))
#endif
        return false;

    return true;
}

void CudaContext::listDevices() {
    int devs = 0;
    CUresult status;
    if ((status = cuDeviceGetCount(&devs)) != CUDA_SUCCESS) {
        logging::error("Unable to get GPU device count");
        return;
    }

    for (int i = 0; i < devs; ++i) {
        CUdevice dev;
        if (cuDeviceGet(&dev, i) != CUDA_SUCCESS) {
            logging::error("Unable to get GPU device %d", i);
            continue;
        }
        char name[256];
        if (cuDeviceGetName(name, sizeof(name), dev) != CUDA_SUCCESS) {
            logging::error("Unable to get name of GPU device %d", i);
            continue;
        }
        logging::info("%d: %s", i, name);
    }
}

// -------------------------------------------------------------------

void CoreRegisters::initialize(bool sim, void* regs)
{
  if (!sim) _fwRegs = std::make_unique<GpuAsyncCoreRegs>(regs);
  else      _swRegs = static_cast<uint8_t*>(regs);
}

// -------------------------------------------------------------------

int CudaContext::getAttribute(CUdevice_attribute attr)
{
    int out;
    if (cuDeviceGetAttribute(&out, attr, device_) == CUDA_SUCCESS)
        return out;
    return 0;
}

// -------------------------------------------------------------------

bool Drp::Gpu::checkError(CUresult status, const char* const func, const char* const file,
                          const int line, const bool crash, const char* const msg)
{
    if (status != CUDA_SUCCESS) {
        const char* perrstr = 0;
        CUresult ok         = cuGetErrorString(status, &perrstr);
        const char* perrnam = 0;
        CUresult ok2        = cuGetErrorName(status, &perrnam);
        const char* message = msg ? msg : ""; // Just in case, but msg is never 0
        if (ok == CUDA_SUCCESS && ok2 == CUDA_SUCCESS) {
            if (perrstr) {
                logging::error("%s:%d:  %s (%i): '%s' %s",
                               file, line, perrnam, status, perrstr, message);
            } else {
                logging::error("%s:%d:  %s (%i): unknown error %s",
                               file, line, perrnam, status, message);
            }
        } else {
            logging::error("%s:%d:  status %i: unknown error %s",
                           file, line, status, message);
        }
        if (crash)  abort();
        return true;
    }
    return false;
}

bool Drp::Gpu::checkError(cudaError status, const char* const func, const char* const file,
                          const int line, const bool crash, const char* const msg)
{
    if (status != cudaSuccess) {
        logging::error("%s:%d:  %s (%i): '%s' %s",
                       file, line, cudaGetErrorName(status), status,
                       cudaGetErrorString(status), msg ? msg : "");
        if (crash)  abort();
        return true;
    }
    return false;
}

// -------------------------------------------------------------------

DmaTgt_t Drp::Gpu::dmaTgtGet(CoreRegisters& coreRegs)
{
    DmaTgt_t tgt;
    switch (coreRegs.axisDeMuxSelect()) {
        case TGT_CPU:  tgt = TGT_CPU;  break;
        case TGT_GPU:  tgt = TGT_GPU;  break;
        default:       tgt = TGT_ERR;  break;
    }
    return tgt;
}

void Drp::Gpu::dmaTgtSet(CoreRegisters& coreRegs, DmaTgt_t tgt)
{
    coreRegs.setAxisDeMuxSelect(tgt);
}

/** Function to reset the DMA buffer index */
void Drp::Gpu::dmaIdxReset(CoreRegisters& coreRegs)
{
    // Toggle the writeEnable register to reset the DMA buffer index
    coreRegs.setWriteEnable(0);
    coreRegs.setWriteEnable(1);
}

std::string Drp::Gpu::pciAddrOfGpu(int device)
{
    // 13 is the minimum for "0000:00:00.0"; CUDA fails rather than truncate
    char pciBusId[16] = {0};
    if (chkError(cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), device)))
        return std::string();

    // CUDA returns the domain upper case, where the sysfs names are lower
    for (auto p = pciBusId; *p; ++p)  *p = tolower(*p);
    return std::string(pciBusId);
}

int Drp::Gpu::numaNodeOfGpu(int device)
{
    auto pciAddr = pciAddrOfGpu(device);
    return pciAddr.empty() ? -1 : Pds::numaNodeOfPciDev(pciAddr);
}

static int                       hostAllocNode = -1;
static std::mutex                hostAllocMutex;
static std::map<void*, size_t>   hostAllocSizes;

void Drp::Gpu::setHostAllocNode(int node)
{
    hostAllocNode = node;
}

cudaError_t Drp::Gpu::hostAlloc(void** buf, size_t size)
{
    const size_t pgSz = sysconf(_SC_PAGESIZE);
    size = ((size ? size : 1) + pgSz - 1) & ~(pgSz - 1);
    *buf = nullptr;

    void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        logging::error("hostAlloc: mmap of %zu B failed: %m", size);
        return cudaErrorMemoryAllocation;
    }
    if (hostAllocNode >= 0) {
        unsigned long mask = 1ul << hostAllocNode;
        if (syscall(SYS_mbind, ptr, size, MPOL_BIND, &mask, 8 * sizeof(mask), MPOL_MF_STRICT) < 0) {
            logging::error("hostAlloc: mbind to NUMA node %d failed: %m", hostAllocNode);
            munmap(ptr, size);
            return cudaErrorMemoryAllocation;
        }
    }
    memset(ptr, 0, size);               // Fault the pages in, under the binding

    auto rc = cudaHostRegister(ptr, size, cudaHostRegisterPortable | cudaHostRegisterMapped);
    if (rc != cudaSuccess) {
        munmap(ptr, size);
        return rc;
    }

    // Callers pass these pointers straight to kernels, as they did cudaHostAlloc()'s
    void* dptr = nullptr;
    rc = cudaHostGetDevicePointer(&dptr, ptr, 0);
    if (rc != cudaSuccess || dptr != ptr) {
        logging::error("hostAlloc: device pointer %p differs from host pointer %p", dptr, ptr);
        cudaHostUnregister(ptr);
        munmap(ptr, size);
        return rc != cudaSuccess ? rc : cudaErrorNotSupported;
    }

    std::lock_guard<std::mutex> lk(hostAllocMutex);
    hostAllocSizes[ptr] = size;
    *buf = ptr;
    return cudaSuccess;
}

cudaError_t Drp::Gpu::hostFree(void* buf)
{
    if (!buf)  return cudaSuccess;
    size_t size;
    {
        std::lock_guard<std::mutex> lk(hostAllocMutex);
        auto it = hostAllocSizes.find(buf);
        if (it == hostAllocSizes.end())  return cudaErrorInvalidValue;
        size = it->second;
        hostAllocSizes.erase(it);
    }
    auto rc = cudaHostUnregister(buf);
    munmap(buf, size);
    return rc;
}

void Drp::Gpu::logHostPlacement(const char* who, const char* what, const void* buf, size_t size, int gpuNode)
{
    const uintptr_t pgSz  = sysconf(_SC_PAGESIZE);
    const uintptr_t first = uintptr_t(buf) & ~(pgSz - 1);
    const uintptr_t last  = (uintptr_t(buf) + (size ? size : 1) - 1) & ~(pgSz - 1);
    std::vector<void*> pages;
    for (auto addr = first; addr <= last; addr += pgSz)  pages.push_back((void*)addr);
    std::vector<int> status(pages.size(), -1);

    // move_pages() with no target nodes only reports where each page is
    if (syscall(SYS_move_pages, 0, pages.size(), pages.data(), nullptr, status.data(), 0) < 0) {
        logging::warning("%s: %s: move_pages failed: %m", who, what);
        return;
    }
    std::map<int, size_t> count;
    for (auto node : status)  ++count[node];
    std::string where;
    size_t remote = 0;
    for (auto& [node, nPages] : count) {
        where += (node < 0 ? " err" + std::to_string(-node) : " node" + std::to_string(node)) + ":" + std::to_string(nPages);
        if (node != gpuNode)  remote += nPages;
    }
    if (remote)
        logging::warning("%s: %s: %zu of %zu pages are not on the GPU's NUMA node %d:%s",
                         who, what, remote, pages.size(), gpuNode, where.c_str());
    else
        logging::info("%s: %s: all %zu pages on the GPU's NUMA node %d",
                      who, what, pages.size(), gpuNode);
}

// -------------------------------------------------------------------
