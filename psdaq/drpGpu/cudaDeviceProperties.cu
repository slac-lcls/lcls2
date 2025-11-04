#include <stdio.h>

#include <cuda.h>

int main() {

  int nDevices;
  cudaGetDeviceCount(&nDevices);

  printf("Number of devices: %d\n", nDevices);

  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  SM capability: %d.%d\n", prop.major, prop.minor);
    printf("  Device Clock Rate (MHz): %d\n", prop.clockRate/1024);
    printf("  Memory Clock Rate (MHz): %d\n", prop.memoryClockRate/1024);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("  Total global memory (GiB): %.1f\n",(float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0);
    printf("  L2 cache size (MiB): %.1f\n", (float)(prop.l2CacheSize)/1024.0/1024.0);
    printf("  Persisting L2 cache size (MiB): %.1f\n", (float)(prop.persistingL2CacheMaxSize)/1024.0/1024.0);
    printf("  L1 cache supported: %s\n", prop.localL1CacheSupported ? "yes" : "no");
    printf("  Registers per Block: %d\n", prop.regsPerBlock);
    printf("  Registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("  Shared memory per SM (KiB): %.1f\n",(float)(prop.sharedMemPerMultiprocessor)/1024.0);
    printf("  Shared memory per block (KiB): %.1f\n",(float)(prop.sharedMemPerBlock)/1024.0);
    printf("  Warp size: %d\n", prop.warpSize);
    printf("  Unified addressing: %s\n", prop.unifiedAddressing ? "yes" : "no");
    printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
    printf("  Concurrent computation/communication: %s\n",prop.deviceOverlap ? "yes" : "no");
    printf("  Compute preemption: %s\n", prop.computePreemptionSupported ? "yes" : "no");
    printf("  Can map host memory: %s\n", prop.canMapHostMemory ? "yes" : "no");
    printf("  Number of multiprocessors: %d\n", prop.multiProcessorCount);
    printf("  Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Max blocks per multiprocessor: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
  }
}
