// vim: et
#include <stdint.h>
#include <string>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <getopt.h>
#include <exception>
#include <getopt.h>
#include <vector>
#include <thread>

#include "GpuAsyncLib.hh"

#include "GpuAsyncOffsets.h"

// Define to disable logging in hot-paths. If you keep logging enabled, latencies will appear longer than they should.
#define PROFILE

//#define CHECK_MEM
#undef PROFILE
#ifndef PROFILE
#define logInfo(...) fprintf(stderr, __VA_ARGS__)
#else
#define logInfo(...)
#endif

#define GPU_BUFFER_SIZE 0x10000

//-----------------------------------------------------------------------------//

struct GpuTestState_t
{
    int fd;

    CUcontext context;

    GpuDmaBuffer_t swFpgaRegs;

    CUstream streams[MAX_BUFFERS];

    GpuBufferState_t buffers[MAX_BUFFERS];
    int buffer_count;

    int iters;

    bool verbose;
};

//-----------------------------------------------------------------------------//

void report_latencies(GpuTestState_t& state, int buffer);
void show_help(const char* av0);
int run_host_test_guts(GpuTestState_t& state, int instance);
int run_host_test(GpuTestState_t& state);

//-----------------------------------------------------------------------------//

int main(int argc, char* argv[]) {
    int res = -1;

    char dev[256] = "/dev/datagpu_0";

    ////////////////////////////////////////////
    // Option parsing
    ////////////////////////////////////////////
    int opt = -1;
    unsigned lverbose = 0;
    int iters = -1, list_devs = 0, buffers = MAX_BUFFERS, gpu = 0;
    bool cpu = false;                   // GPU mode
    while ((opt = getopt(argc, argv, "hc:d:lb:g:Cv")) != -1) {
        switch(opt) {
        case 'd':
            strcpy(dev, optarg);
            break;
        case 'c':
            iters = atoi(optarg);
            break;
        case 'l':
            list_devs = 1;
            break;
        case 'b':
            buffers = atoi(optarg);
            break;
        case 'g':
            gpu = atoi(optarg);
            break;
        case 'v':
            ++lverbose;
            break;
        case 'C':
            cpu = true;
            break;
        case 'h':
        default:
            show_help(argv[0]);
        }
    }

    ////////////////////////////////////////////
    // Validate arguments
    ////////////////////////////////////////////

    if (buffers <= 0 || buffers > MAX_BUFFERS) {
        fprintf(stderr, "-b must be in range 0 < x <= MAX_BUFFERS (%d)\n", MAX_BUFFERS);
        return 1;
    }

    ////////////////////////////////////////////
    // Open and setup GPU
    ////////////////////////////////////////////

    printf("Opening device %s\n", dev);
    DataGPU gpu0(dev);

    CudaContext context;

    if (list_devs) {
        context.listDevices();
        return 0;
    }

    if (!context.init(gpu)) {
        fprintf(stderr, "CUDA init failed\n");
        return 1;
    }

    int value;
    cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, context.device());
    if (lverbose)  printf("Device supports unified addressing: %s\n", value ? "YES" : "NO");

    // Make the DMA target the GPU
    auto tgt = dmaTgtGet(gpu0.fd());
    const char* tgtName = "";
    switch (tgt) {
        case DmaTgt_t::CPU:  tgtName = "CPU";  break;
        case DmaTgt_t::GPU:  tgtName = "GPU";  break;
        default:             tgtName = "ERR";  break;
    };
    if (lverbose)  printf("DMA target is changing from %s to %s\n",
                          tgtName, cpu ? "CPU" : "GPU");
    if (!cpu && tgt != GPU)
        dmaTgtSet(gpu0.fd(), DmaTgt_t::GPU);
    else if (cpu && tgt != CPU)
        dmaTgtSet(gpu0.fd(), DmaTgt_t::CPU);

    ////////////////////////////////////////////////
    // Create write and read buffers
    ////////////////////////////////////////////////

    // This will store everything important for now
    GpuTestState_t state {};

    state.fd = gpu0.fd();
    state.context = context.context();
    state.buffer_count = buffers;
    state.verbose = lverbose;
    state.iters = iters;

    // Allocate a buffer list for our data
    // This handles allocating buffers on the device and registering them with the driver.
    // After this call, the FPGA will be properly configured to run the test code.
    for ( int i = 0; i < buffers; ++i ) {
        if (gpuInitBufferState(&state.buffers[i], gpu0, GPU_BUFFER_SIZE) < 0) {
            fprintf(stderr, "Failed to alloc buffer list\n");
            exit(1);
        }
    }

    ////////////////////////////////////
    // Allocate stream and run tests
    ////////////////////////////////////

    if (state.verbose)  logInfo("Create streams\n");

    /** Allocate a stream per buffer **/
    for (int i = 0; i < state.buffer_count; ++i) {
        if (cudaStreamCreate(&state.streams[i]) != cudaSuccess) {
            fprintf(stderr, "Error creating streams\n");
            return 1;
        }
    }

    int ret = run_host_test(state);

    ////////////////////////////////////
    // Final cleanup
    ////////////////////////////////////

    for (int i = 0; i < state.buffer_count; ++i)
        gpuDestroyBufferState(&state.buffers[i]);

    res = gpuRemNvidiaMemory(gpu0.fd());
    if (res < 0) fprintf(stderr, "Error in IOCTL_GPUDMA_MEM_UNLOCK\n");

    return ret;
}

int run_host_test(GpuTestState_t& state)
{
    std::vector<std::thread> threads;
    for (int i = 0; i < state.buffer_count; ++i) {
        threads.emplace_back([&state, i]{
            run_host_test_guts(state, i);
        });
    }

    for (int i = 0; i < state.buffer_count; ++i) {
        threads[i].join();
    }

    return 0;
}

static void show_buf(CUdeviceptr dptr, size_t size, CUstream stream = 0) {
    uint8_t buf[size];
    //cuMemcpyDtoH(buf, dptr, size);
    chkFatal(cudaMemcpyAsync(buf, (void*)dptr, size, cudaMemcpyDeviceToHost, stream));
    cuStreamSynchronize(stream);
    for (unsigned i = 0; i < size / 4; ++i) {
        printf("%2d: offset=0x%X,  0x%X\n",i,i*4,*((uint32_t*)(buf+(i*4))));
    }
}

/*******************************************************************
 * Implementation of the HOST test
 * \return -1 on error, 0 on success
 *******************************************************************/
int run_host_test_guts(GpuTestState_t& state, int instance)
{
    const auto& stream = state.streams[instance];
    auto  hwWritePtr   = state.buffers[instance].bwrite.dptr;
    auto  iters        = state.iters;

    while(iters < 0 || iters-- > 0) {
        // Clear the GPU memory handshake space to zero
        if (state.verbose)  logInfo("%d Clear memory\n", instance);
        chkFatal(cuStreamWriteValue32(stream, hwWritePtr + 4, 0x00, 0));

        // Write to the DMA start register in the FPGA
        if (state.verbose)  logInfo("%d Trigger write\n", instance);
        //chkFatal(cuStreamWriteValue32(stream, state.hwWriteStart + 4 * instance, 0x01, 0));
        cuStreamSynchronize(stream);
        auto rc = gpuSetWriteEn(state.fd, instance);
        if (rc < 0) {
          logInfo("Failed to reenable buffer %d for write: %zd\n", instance, rc);
          perror("gpuSetWriteEn");
          abort();
        }

#ifdef CHECK_MEM
        //while(1)
        {
            cuStreamSynchronize(stream);
            show_buf(hwWritePtr, 0x100);
        }
#endif

        // Spin on the handshake location until the value is greater than or equal to 1
        // This waits for the data to arrive before starting the processing
        if (state.verbose)  logInfo("%d Wait memory value\n", instance);
        chkFatal(cuStreamWaitValue32(stream, hwWritePtr + 4, 0x1, CU_STREAM_WAIT_VALUE_GEQ));

        {
            cuStreamSynchronize(stream);
            if (state.verbose)  logInfo("%d Dump:\n", instance);
            show_buf(hwWritePtr, 2 * 8 * sizeof(uint32_t), stream);
        }
    }

    return 0;
}


void show_help(const char* av0)
{
    printf("USAGE: %s [-t type] [-h] [-l] [-g gpu_number] [-b buff_count]\n", av0);
    printf("   -d <dev>    : The device to use. Defaults to /dev/datagpu_0\n");
    printf("   -h          : Show this help text\n");
    printf("   -l          : List the available GPUs\n");
    printf("   -g <number> : Select the specified GPU\n");
    printf("   -c <iters>  : Do this many iterations of the test\n");
    printf("   -b <count>  : Allocate this many buffers for round robin operation. Must be 0 < X <= MAX_BUFFERS (%d)\n", MAX_BUFFERS);
    printf("   -C          : Force the DMA target to be the CPU (defaults to GPU)\n");
    exit(0);
}
