// This header is NOT safe to include in CPU code.

#pragma once

#include "ReducerAlgo.hh"

#include "RingQueue_HtoD.hh"
#include "RingQueue_DtoH.hh"


namespace Drp {
  namespace Gpu {

static __device__ bool     indexValid  = false;
static __device__ unsigned blockCount1 = 0;
static __shared__ bool     isLastBlockDone1;
static __device__ unsigned blockCount2 = 0;
static __shared__ bool     isLastBlockDone2;

// GPU kernel for actually performing the data reduction
template<typename ReduceFn>
static inline __global__
void _reduce(unsigned*                    const __restrict__ index,
             RingQueueHtoD<unsigned>*     const __restrict__ inputQueue,
             ReduceFn                                        reduceFn,
             float     const*             const __restrict__ calibBuffers,
             size_t    const                                 calibBufsCnt,
             uint8_t*                     const __restrict__ dataBuffers,
             size_t    const                                 dataBufsCnt,
             float     const*             const __restrict__ refBuffers,
             unsigned  const                                 refBufCnt,
             unsigned*                    const __restrict__ error,
             RingQueueDtoH<ReducerTuple>* const __restrict__ outputQueue,
             unsigned*                    const __restrict__ done)
{
  if (threadIdx.x == 0) {                  // Thread 0 of each block
    if (blockIdx.x == 0) {
      // Find which pebble buffers are ready for processing
      //receive(index, inputQueue, done);
      //printf("### _reducer: pop index\n");
      *done |= !inputQueue->pop(index);
      //printf("### _reducer: idx %u\n", *index);
    }

    __threadfence();                                     // Ensure global memory is updated before the following
    unsigned value = atomicInc(&blockCount1, gridDim.x); // Thread 0 signals that it is done
    isLastBlockDone1 = (value == (gridDim.x - 1));       // Thread 0 determines if its block is the last block to be done
  }
  __syncthreads();   // Synchronize to ensure that each thread (all blocks) reads the correct value of isLastBlockDone
  if (isLastBlockDone1) {               // Only last block will have set isLastBlockDone true
    if (threadIdx.x == 0) {             // Thread 0 updates global memory
      indexValid = true;                // Thread 0 of last block signals index in global memory is valid
      blockCount1 = 0;                  // Reset for next time
    }
  }
  unsigned ns{8};
  while(!indexValid) {                  // All threads wait for index to become valid
    //if (terminate)  return;
    __nanosleep(ns);
    if (ns < 256)  ns *= 2;
  }
  if (*done)  return;

  // Perform the reduction algorithm
  reduceFn(*index, calibBuffers, calibBufsCnt, dataBuffers, dataBufsCnt, refBuffers, refBufCnt, error);

  __syncthreads();    // Wait for all threads to complete so that thread 0 can't increment blockCount before they're done
  if (threadIdx.x == 0) {
    __threadfence();                                     // Ensure global memory is updated before the following
    unsigned value = atomicInc(&blockCount2, gridDim.x); // Thread 0 signals that it is done
    isLastBlockDone2 = (value == (gridDim.x - 1));       // Thread 0 determines if its block is the last  block to be done
  }
  __syncthreads();    // Synchronize to make sure that each thread reads the correct value of isLastBlockDone
  if (isLastBlockDone2) {
    if (threadIdx.x == 0) {
      // Re-launch! Additional behavior can be put in graphLoop as needed. For now, it just re-launches the current graph.
      //graphLoop(index, dataBuffers, dataBufsCnt, outputQueue, done);
      auto const __restrict__ data = &dataBuffers[*index * dataBufsCnt];
      auto dataSize = ((size_t*)data)[-1];
      //printf("### Reducer graphLoop: push {%u, %lu}, done %u\n", *index, dataSize, *done);
      *done |= !outputQueue->push({*index, dataSize});
      blockCount2 = 0;                // Reset for next time
      indexValid = false;             // syncthreads() happens after DMA on next launch
      if (!*done) {
        //printf("### _reducer: relaunch\n");
        cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
      }
    }
  }
}

  } // Gpu
} // Drp
