// The Reader's per-event kernel, as a template on a detector-supplied policy.
//
// Why a template rather than a virtual method or a device function pointer:
//
// The DMAed data is a bucket of bytes whose meaning is known only to the
// detector, whose code lives in a dynamically loaded .so.  The kernel that
// processes it lives in the drp_gpu executable.  Those are separate CUDA
// modules, and device code cannot be linked across them: device linking
// (-rdc=true plus -dlink) happens at build time within one link unit, and a
// dlopen'd .so is a different link unit.  Consequences, all measured on an
// H200 with CUDA 13.3:
//
// - Taking &someDeviceFn in host code yields a *host* address.  Calling it from
//   a kernel gives CUDA_EXCEPTION_23 (warp misaligned PC) or an illegal memory
//   access, depending on what the garbage points at.
// - Capturing the pointer properly, in device code, and calling it from a kernel
//   in the other module does currently work, but it is outside the programming
//   model: nothing guarantees that address stays valid across a driver or
//   toolkit upgrade, a dlclose, or CUDA's lazy module loading.  It also costs
//   19% on a full EpixUHR3x2 frame, because an indirect call requires
//   -rdc=true, which forces the ABI calling convention and blocks inlining.
//   That is the same mechanism a virtual method would use, so it buys nothing.
//
// Making the kernel a template moves the polymorphism to the host.  Each
// detector .so instantiates the kernel with its own policy, so the policy is
// compiled into the same module as the kernel and inlines into it.  The only
// virtual call is Gpu::Detector::recordEvent(), on the host, once per graph
// recording.
//
// A policy supplies:
//   __device__ void process(const EventPayload&, unsigned tid, unsigned stride) const
// and owns the interpretation of the bucket of bytes completely: the element
// type, whether pedestals and gains are involved, and where in the calibrated
// buffer each piece of the frame belongs.  That last point matters: sub-frame
// tdests are not necessarily a contiguous run (Jungfrau puts module 0 on tdest
// 2 and the rest on 4, 5, 6, ...), and a batch may be nested (each Jungfrau
// module's sub-frame is itself a batch of UDP packets).  EvtBatcherIterator is
// __device__ callable, so a policy needing a second-level walk can do one.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "MemPool.hh"                   // For DmaDsc
#include "EventBatcher.hh"
#include "psdaq/service/EbDgram.hh"     // For Pds::TimingHeader

namespace Drp {
  namespace Gpu {

// What the _event kernel needs that no detector has to know about.  Assembled by
// Reader::_recordGraph and handed to Gpu::Detector::recordEvent().
struct EventKernelArgs
{
  unsigned                     reader;
  unsigned*                    state;
  unsigned*                    dmaBufferIdx;
  unsigned*                    pebbleIdx;
  uint8_t const* const*        dmaBuffers;    // [dmaCount][maxDmaSize]
  size_t                       frameSize;     // Bytes of the line bearing the DmaDsc
  uint32_t*                    hdrBuffers;    // [nBuffers * hdrBufsCnt]
  size_t                       hdrBufsCnt;
  float*                       calibBuffers;  // [nBuffers * calibBufsCnt]
  size_t                       calibBufsCnt;
  EvtBatcherSubFrames const*   subFrames;     // nullptr when the data isn't batched
  uint64_t*                    stateMon;
};

// This event's bucket of bytes, and where its results go.  Handed to the policy.
struct EventPayload
{
  uint8_t const*             data;       // The DMA payload, past the DmaDsc line
  size_t                     size;       // DmaDsc::size, i.e. bytes of 'data'
  float*                     out;        // This event's calibrated buffer
  size_t                     outCnt;     // Elements available in 'out'
  EvtBatcherSubFrames const* subFrames;  // The cached scan; null when not batched
  bool                       batched;    // False for a transition, whose payload
                                         // is a bare TimingHeader
  unsigned                   pebbleIdx;  // For indexing per-event side buffers
};

// The generic part of handling an event: publish the DmaDsc and TimingHeader to
// the host-visible buffer, then let the detector's policy interpret the rest.
//
// 'calib' is passed by value so that a policy can carry whatever constants and
// device pointers it needs.
template<class Calib>
__global__
void _event(EventKernelArgs const a, Calib const calib)
{
  if (*a.state != 2)  return;

  auto const tid    = blockIdx.x * blockDim.x + threadIdx.x;
  auto const stride = blockDim.x * gridDim.x;

  auto const dmaBufIdx{*a.dmaBufferIdx}; // All threads load these into a register
  auto const pblBufIdx{*a.pebbleIdx};    // from global memory

  auto const __restrict__ in  = (uint32_t const*)a.dmaBuffers[dmaBufIdx];
  auto const __restrict__ hdr = a.hdrBuffers + pblBufIdx * a.hdrBufsCnt;

  constexpr auto nDscWds = sizeof(DmaDsc)/sizeof(uint32_t);
  constexpr auto nHdrWds = sizeof(Pds::TimingHeader)/sizeof(uint32_t);
  constexpr auto nLdrWds = nDscWds + nHdrWds;
  auto const     nFrmWds = a.frameSize/sizeof(uint32_t);

  // Where the TimingHeader sits depends on whether the Detector presents
  // sub-frames: without them the payload starts with it; with them the payload
  // starts with the batch and it is sub-frame 0, as on the CPU side.  Either way
  // a transition's payload is a bare TimingHeader, which is what tells the two
  // apart.
  auto const              dmaSize = in[1];
  auto const              batched = a.subFrames && (dmaSize != sizeof(Pds::TimingHeader));
  auto const __restrict__ payload = (uint8_t const*)&in[nFrmWds];
  auto const __restrict__ th      = (uint32_t const*)(batched ? (*a.subFrames)[0].data(payload)
                                                              : payload);
  if      (tid < nDscWds)  { hdr[tid] = in[tid]; }
  else if (tid < nLdrWds)  { hdr[tid] = th[tid - nDscWds]; }

  EventPayload const p{payload,
                       dmaSize,
                       &a.calibBuffers[pblBufIdx * a.calibBufsCnt],
                       a.calibBufsCnt,
                       a.subFrames,
                       batched,
                       pblBufIdx};
  calib.process(p, tid, stride);

  // The state variable is likely set before the last thread is done, but the
  // next kernel won't check it before all threads of this kernel complete
  if (tid == 0)  *a.state = 3;
}

// Grid-stride pedestal/gain calibration of one contiguous run of raw elements.
//
// The pedestal and gain arrays are laid out as [range][pgStride], covering the
// detector's whole frame, so a caller calibrating the frame a piece at a time
// passes the frame's element count as pgStride and the piece's position within
// the frame as pgOffset.  Handling the frame in one go means pgStride == the
// frame's element count and pgOffset == 0.
__device__
inline void pedGainCalibrate(float*        const        __restrict__ calib,
                             uint16_t      const* const __restrict__ raw,
                             unsigned      const                     nElements,
                             unsigned      const                     rangeOffset,
                             unsigned      const                     rangeBits,
                             float         const* const __restrict__ pedArray,
                             float         const* const __restrict__ gainArray,
                             unsigned      const                     pgStride,
                             unsigned      const                     pgOffset,
                             float         const* const __restrict__ ref,
                             unsigned      const                     tid,
                             unsigned      const                     stride)
{
  auto const rangeMask{(1u << rangeBits) - 1u};
  auto const dataMask {(1u << rangeOffset) - 1u};
  for (auto i = tid; i < nElements; i += stride) {
    auto const              range = (raw[i] >> rangeOffset) & rangeMask;
    auto const __restrict__ peds  = &pedArray [range * pgStride + pgOffset];
    auto const __restrict__ gains = &gainArray[range * pgStride + pgOffset];
    auto const              data  = raw[i] & dataMask;
    calib[i] = (float(data) - peds[i]) * gains[i];

    //if (ref && (calib[i] != ref[i])) {
    //  printf("### Reader: blk %d, thr %d, Mismatch @ %u: calib %f != ref %f\n",
    //         blockIdx.x, threadIdx.x, i, calib[i], ref[i]);
    //}
  }
}

// The policy for detectors whose payload is one contiguous block of uint16_t
// following the TimingHeader, calibrated on the GPU from pedestals and gains:
// AreaDetector, EpixUHRemu and EpixUHRsim.
struct PedGainCalib
{
  float const* peds;
  float const* gains;
  float const* ref;                     // Simulator mode only; null otherwise
  unsigned     refBufCnt;
  unsigned     rangeOffset;
  unsigned     rangeBits;

  __device__
  void process(const EventPayload& p, unsigned tid, unsigned stride) const
  {
    if (p.size <= sizeof(Pds::TimingHeader))  return;   // Transition: no payload

    auto const __restrict__ raw = (uint16_t const*)(p.data + sizeof(Pds::TimingHeader));
    auto const payloadCnt = (p.size - sizeof(Pds::TimingHeader))/sizeof(uint16_t);
    auto const elementCnt = payloadCnt > p.outCnt ? p.outCnt : payloadCnt;
    auto const __restrict__ r = ref && refBufCnt
                              ? &ref[(p.pebbleIdx % refBufCnt) * p.outCnt]
                              : (float const*)nullptr;
    // pgStride is the pedestal/gain plane stride, i.e. the detector's frame
    // size, not this event's element count, which may be short
    pedGainCalibrate(p.out, raw, elementCnt, rangeOffset, rangeBits,
                     peds, gains, p.outCnt, 0, r, tid, stride);
  }
};

  } // Gpu
} // Drp
