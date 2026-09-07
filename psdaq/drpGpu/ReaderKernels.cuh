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

// Per-event status, handed to the host in the last word of the event's
// host-visible buffer.  A kernel can neither log nor throw, and device-side
// printf is for debugging only, so the Reader records a code and
// TrgInpGen::_receiver() does the reporting, alongside the other data integrity
// checks it already makes.
enum EventStatus : uint32_t {
  EventStatusOk = 0,
  EventStatusDmaSizeTooSmall,           // DMA shorter than a TimingHeader
  EventStatusBatchUnintelligible,       // Sub-frame scan failed; the reason is in
                                        // Reader::batcherStatus()
};

// Where that code sits within one event's host-visible buffer
__host__ __device__
inline size_t eventStatusIndex(size_t hdrBufsCnt) { return hdrBufsCnt - 1; }

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
  unsigned const*              evtStatus;     // An EventStatus, set by _waitForDMA
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
  bool                       batched;    // The Detector presents sub-frames
  bool                       hasData;    // This event bears detector data, which
                                         // the cached scan describes.  False for
                                         // a transition, and for a payload whose
                                         // size or layout was not understood.
  unsigned                   pebbleIdx;  // For indexing per-event side buffers
};

// The generic part of handling an event: publish the DmaDsc and TimingHeader to
// the host-visible buffer, then let the detector's policy interpret the rest.
//
// 'calib' is passed by value so that a policy can carry whatever constants and
// device pointers it needs.
template<class Calib>
__global__
void _event(EventKernelArgs const args, Calib const calib)
{
  if (*args.state != 2)  return;

  auto const tid    = blockIdx.x * blockDim.x + threadIdx.x;
  auto const stride = blockDim.x * gridDim.x;

  auto const dmaBufIdx{*args.dmaBufferIdx}; // All threads load these into a register
  auto const pblBufIdx{*args.pebbleIdx};    // from global memory

  auto const __restrict__ in  = (uint32_t const*)args.dmaBuffers[dmaBufIdx];
  auto const __restrict__ hdr = args.hdrBuffers + pblBufIdx * args.hdrBufsCnt;

  constexpr auto nDscWds = sizeof(DmaDsc)/sizeof(uint32_t);
  constexpr auto nHdrWds = sizeof(Pds::TimingHeader)/sizeof(uint32_t);
  constexpr auto nLdrWds = nDscWds + nHdrWds;
  auto const     nFrmWds = args.frameSize/sizeof(uint32_t);

  // Where the TimingHeader sits depends on whether the Detector presents
  // sub-frames.  Without them, the payload starts with it.  With them, the
  // payload starts with the batch and the TimingHeader is sub-frame 0's data,
  // which is the line straight after the batcher header -- exactly what
  // Drp::BEBDetector::getTimingHeader() computes with ebh->next().  Note that
  // that is done for every DMA, transitions included: a Detector that batches
  // batches its transitions too, and a transition's batch is simply one with no
  // data sub-frames.  Deriving the TimingHeader from the header's line width
  // rather than from the cached scan therefore works for both, and needs no scan.
  auto const              dmaSize = in[1];
  auto const              batched = args.subFrames != nullptr;
  auto const __restrict__ payload = (uint8_t const*)&in[nFrmWds];
  auto const __restrict__ th      = (uint32_t const*)
    (batched ? payload + ((EvtBatcherHeader const*)payload)->lineWidth()
             : payload);
  // Whether there is anything for the policy to interpret.  For a batched
  // Detector that means the cached scan describes this payload; otherwise it
  // means the payload holds more than just the TimingHeader.
  auto const hasData = batched ? args.subFrames->matches(dmaSize)
                               : dmaSize > sizeof(Pds::TimingHeader);
  if      (tid < nDscWds)   { hdr[tid] = in[tid]; }
  else if (tid < nLdrWds)   { hdr[tid] = th[tid - nDscWds]; }
  // Pass _waitForDMA's verdict on this event to the host to report
  else if (tid == nLdrWds)  { hdr[eventStatusIndex(args.hdrBufsCnt)] = *args.evtStatus; }

  EventPayload const pyld{payload,
                          dmaSize,
                          &args.calibBuffers[pblBufIdx * args.calibBufsCnt],
                          args.calibBufsCnt,
                          args.subFrames,
                          batched,
                          hasData,
                          pblBufIdx};
  calib.process(pyld, tid, stride);

  // The state variable is likely set before the last thread is done, but the
  // next kernel won't check it before all threads of this kernel complete
  if (tid == 0)  *args.state = 3;
}

// Grid-stride pedestal/gain calibration of one contiguous run of raw elements.
//
// The pedestal and gain arrays are laid out as [range][pgStride], covering the
// detector's whole frame, so a caller calibrating the frame a piece at a time
// passes the frame's element count as pgStride and the piece's position within
// the frame as pgOffset.  Handling the frame in one go means pgStride == the
// frame's element count and pgOffset == 0.
__device__
inline void pedGainCalibrate(float*   const        __restrict__ calib,
                             uint16_t const* const __restrict__ raw,
                             unsigned const                     nElements,
                             unsigned const                     rangeOffset,
                             unsigned const                     rangeBits,
                             float    const* const __restrict__ pedArray,
                             float    const* const __restrict__ gainArray,
                             unsigned const                     pgStride,
                             unsigned const                     pgOffset,
                             float    const* const __restrict__ ref,
                             unsigned const                     tid,
                             unsigned const                     stride)
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
  void process(const EventPayload& pyld, unsigned tid, unsigned stride) const
  {
    if (!pyld.hasData)  return;         // A transition, or nothing intelligible

    auto const __restrict__ raw = (uint16_t const*)(pyld.data + sizeof(Pds::TimingHeader));
    auto const payloadCnt = (pyld.size - sizeof(Pds::TimingHeader))/sizeof(uint16_t);
    auto const elementCnt = payloadCnt > pyld.outCnt ? pyld.outCnt : payloadCnt;
    auto const __restrict__ refBuf = ref && refBufCnt
                                   ? &ref[(pyld.pebbleIdx % refBufCnt) * pyld.outCnt]
                                   : (float const*)nullptr;
    // pgStride is the pedestal/gain plane stride, i.e. the detector's frame
    // size, not this event's element count, which may be short
    pedGainCalibrate(pyld.out, raw, elementCnt, rangeOffset, rangeBits,
                     peds, gains, pyld.outCnt, 0, refBuf, tid, stride);
  }
};

  } // Gpu
} // Drp
