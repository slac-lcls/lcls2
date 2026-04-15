// This header is NOT safe to include in CPU code.

#pragma once

#include "ReducerAlgo.hh"

#include "RingQueue_HtoD.hh"
#include "RingQueue_DtoH.hh"


namespace Drp {
  namespace Gpu {

static __device__ bool     done1          = false;
static __device__ unsigned blockCount1    = 0;
static __shared__ bool     isLastBlockDone1;
static __device__ bool     lastBlockFlag1 = false;
static __device__ bool     done2          = false;
static __device__ unsigned blockCount2    = 0;
static __shared__ bool     isLastBlockDone2;
static __device__ bool     lastBlockFlag2 = false;
static __device__ bool     done3          = false;
static __device__ unsigned blockCount3    = 0;
static __shared__ bool     isLastBlockDone3;
static __device__ bool     lastBlockFlag3 = false;

static __device__
bool _pop(unsigned*                const __restrict__ index,
          RingQueueHtoD<unsigned>* const __restrict__ inputQueue)
{
  unsigned ns{8};
  while (!inputQueue->pop(index)) {
    __nanosleep(ns);
    if (ns < 256)  ns *= 2;
    else return false;
  }
  return true;
}

static __device__
bool _push(unsigned*                    const __restrict__ index,
           RingQueueDtoH<ReducerTuple>* const __restrict__ outputQueue,
           uint8_t*                     const __restrict__ data,
           size_t                       const              size)
{
  auto dataSize = &((size_t*)data)[-1];
  *dataSize = size;
  //printf("### Reducer _send: push {%u, %lu}\n", *index, dataSize);
  unsigned ns{8};
  while (!outputQueue->push({*index, *dataSize})) {
    __nanosleep(ns);
    if (ns < 256)  ns *= 2;
    else return false;
  }
  return true;
}

static __device__ unsigned lState = 0;

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
             uint64_t*                    const __restrict__ state,
             cuda::std::atomic<unsigned>  const&             terminate_d)
{
  auto const tid = blockIdx.x * blockDim.x + threadIdx.x;

  switch (lState) {
    case 0: {
      if (threadIdx.x == 0) {           // Thread 0 of each block
        if (blockIdx.x == 0) {
          *state = 1;
          // Find which pebble buffers are ready for processing
          done1 = _pop(index, inputQueue);
        }

        __threadfence();                                     // Ensure global memory is updated before the following
        unsigned value = atomicInc(&blockCount1, gridDim.x); // Thread 0 signals that it is done
        isLastBlockDone1 = (value == (gridDim.x - 1));       // Thread 0 determines if its block is the last block to be done
      }
      if (tid == 0)  *state = 2;
      __syncthreads();                  // Synchronize to ensure that each thread (all blocks)
                                        //   reads the correct value of isLastBlockDone
      if (tid == 0)  *state = 3;
      if (isLastBlockDone1) {           // Only last block will have set isLastBlockDone true
        if (threadIdx.x == 0) {         // Thread 0 of last block updates global memory
          lastBlockFlag1 = true;        // Thread 0 of last block signals to all threads it is done
          blockCount1 = 0;              // Reset for next time
        }
      }
      if (tid == 0)  *state = 4;
      __syncthreads();                  // Synchronize to ensure that each thread (all blocks)
                                        //   reads the correct value of lastBlockFlag
      if (tid == 0)  *state = 5;
      unsigned ns{8};
      while(!lastBlockFlag1) {          // All threads wait for last block to have finished
        __nanosleep(ns);
        if (ns < 256)  ns *= 2;
      }
      lastBlockFlag1 = false;           // Reset for next time
      if (!done1)  break;               // No valid index, so relaunch to retry
      if (tid == 0)  *state = 6;
      if (tid == 0)  lState = 1;
      // Fall through to case 1
    }
    case 1: {
      if (tid == 0)  *state = 7;
      // Perform the reduction algorithm
      reduceFn(*index, calibBuffers, calibBufsCnt, dataBuffers, dataBufsCnt, refBuffers, refBufCnt, error);

      if (tid == 0)  *state = 8;
      __syncthreads();                  // Wait for all threads to complete so that thread 0
                                        //   can't increment blockCount before they're done
      if (tid == 0)  *state = 9;
      if (threadIdx.x == 0) {           // Thread 0 of each block
        __threadfence();                                     // Ensure global memory is updated before the following
        unsigned value = atomicInc(&blockCount2, gridDim.x); // Thread 0 signals that it is done
        isLastBlockDone2 = (value == (gridDim.x - 1));       // Thread 0 determines if its block is the last block to be done
      }
      if (tid == 0)  *state = 10;
      __syncthreads();                  // Synchronize to make sure that each thread (all blocks)
                                        //   reads the correct value of isLastBlockDone
      if (tid == 0)  *state = 11;
      if (isLastBlockDone2) {           // Only last block will have set isLastBlockDone true
        if (threadIdx.x == 0) {         // Thread 0 of last block updates global memory
          lastBlockFlag2 = true;        // Thread 0 of last block signals to all threads it is done
          blockCount2 = 0;              // Reset for next time
          auto const __restrict__ data = &dataBuffers[*index * dataBufsCnt];
          auto const dataSize = calibBufsCnt * sizeof(float); // @todo: Needs to come from reduceFn
          done2 = _push(index, outputQueue, data, dataSize);
        }
      }
      if (tid == 0)  *state = 12;
      __syncthreads();                  // Synchronize to ensure that each thread (all blocks)
                                        //   reads the correct value of lastBlockFlag
      if (tid == 0)  *state = 13;
      unsigned ns{8};
      while(!lastBlockFlag2) {          // All threads wait for last block to have finished
        __nanosleep(ns);
        if (ns < 256)  ns *= 2;
      }
      lastBlockFlag2 = false;           // Reset for next time
      if (!done2) {
        if (tid == 0)  *state = 14;
        if (tid == 0)  lState = 2;
        break;                          // Send failed, so retry after relaunch
      }
      if (tid == 0)  *state = 15;
      if (tid == 0)  lState = 0;
      break;
    }
    case 2: {
      if (threadIdx.x == 0) {           // Thread 0 of each block
        if (blockIdx.x == 0) {
          *state = 16;
          auto const __restrict__ data = &dataBuffers[*index * dataBufsCnt];
          auto const dataSize = calibBufsCnt * sizeof(float); // @todo: Needs to come from reduceFn
          done3 = _push(index, outputQueue, data, dataSize);
        }
        __threadfence();                                     // Ensure global memory is updated before the following
        unsigned value = atomicInc(&blockCount3, gridDim.x); // Thread 0 signals that it is done
        isLastBlockDone3 = (value == (gridDim.x - 1));       // Thread 0 determines if its block is the last block to be done
      }
      if (tid == 0)  *state = 17;
      __syncthreads();                  // Synchronize to ensure that each thread (all blocks)
                                        //   reads the correct value of lastBlockFlag
      if (tid == 0)  *state = 18;
      if (isLastBlockDone3) {           // Only last block will have set isLastBlockDone true
        if (threadIdx.x == 0) {         // Thread 0 of last block updates global memory
          lastBlockFlag3 = true;        // Thread 0 of last block signals to all threads it is done
          blockCount3 = 0;              // Reset for next time
        }
      }
      if (tid == 0)  *state = 19;
      __syncthreads();                  // Synchronize to ensure that each thread (all blocks)
                                        //   reads the correct value of lastBlockFlag
      if (tid == 0)  *state = 20;
      unsigned ns{8};
      while(!lastBlockFlag3) {          // All threads wait for last block to have finished
        __nanosleep(ns);
        if (ns < 256)  ns *= 2;
      }
      lastBlockFlag3 = false;           // Reset for next time
      if (tid == 0) {
        *state = 21;
        if (done3)  lState = 0;         // Send completed: go on to next event after relaunch
        *state = 22;
      }
      break;
    }
    default: {
      printf("### _reduce: Illegal state %u\n", lState); // @todo: Replace with error return code
      break;
    }
  }

  // Relaunch the graph
  if (tid == 0) {                       // @todo: Not necessarily 0
    if (!terminate_d.load(cuda::std::memory_order_acquire)) {
      //printf("### _reduce: relaunch\n");
      cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
    }
  }
}

  } // Gpu
} // Drp
