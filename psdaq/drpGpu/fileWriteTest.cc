#include <getopt.h>
#include <limits.h>
#include <unistd.h>
#include <fcntl.h>
#include <fstream>
#include <chrono>
#include <vector>
#include <cstdint>
#include <numeric>
#include <random>
#include <algorithm>
#include <iostream>
#include <string>
#include <cassert>

#include <cuda_runtime.h>

#include "psalg/utils/SysLog.hh"
#include "GpuAsyncLib.hh"
#include "FileWriter.hh"

using logging = psalg::SysLog;
using namespace Drp::Gpu;

static const size_t kB = 1024;
static const size_t MB = 1024 * kB;
static const size_t GB = 1024 * MB;


std::vector<uint64_t> GenerateData(size_t bytes)
{
    assert(bytes % sizeof(uint64_t) == 0);
    std::vector<uint64_t> data(bytes / sizeof(uint64_t));
    std::iota(data.begin(), data.end(), 0);
    std::shuffle(data.begin(), data.end(), std::mt19937{ std::random_device{}() });

    //int fd = open("/dev/urandom", O_RDONLY);
    //read(fd, &data[0], bytes);
    //close(fd);

    return data;
}

long long option_1(const void* data_d,
                   const std::string& filename,
                   size_t recSize,
                   size_t bufSize,
                   unsigned tmoS,
                   size_t& totSize)
{
    const bool dio = true;               // Use Direct IO
    Drp::Gpu::FileWriter fileWriter(recSize, dio);
    XtcData::TimeStamp ts(0, 0);

    unlink(filename.c_str());           // Delete file

    if (fileWriter.open(filename) != 0)  throw "open failed";
    fileWriter.writeEvent(data_d, bufSize, ts); // Warm up
    auto startTime = std::chrono::high_resolution_clock::now();
    auto endTime   = std::chrono::high_resolution_clock::now();
    auto dt        = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    auto tmo       = std::chrono::duration<int, std::micro>(tmoS * 1000000).count();
    totSize   = 0;
    startTime = std::chrono::high_resolution_clock::now();
    do
    {
      fileWriter.writeEvent(data_d, bufSize, ts);
      totSize += bufSize;

      endTime = std::chrono::high_resolution_clock::now();
      dt      = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    } while (dt < tmo);  // Run for awhile to see several points in grafana
    fileWriter.close();

    endTime = std::chrono::high_resolution_clock::now();
    dt      = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

    return dt;
}

long long option_2(const void* data_d,
                   const std::string& filename,
                   size_t recSize,
                   size_t bufSize,
                   unsigned tmoS,
                   size_t& totSize)
{
    const bool dio = true;               // Use Direct IO
    Drp::Gpu::FileWriterAsync fileWriter(recSize, dio);
    XtcData::TimeStamp ts(0, 0);

    unlink(filename.c_str());           // Delete file

    if (fileWriter.open(filename) != 0)  throw "open failed";
    cudaStream_t stream;
    chkFatal(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    fileWriter.registerStream(stream);
    fileWriter.writeEvent(data_d, bufSize, ts); // Warm up
    auto startTime = std::chrono::high_resolution_clock::now();
    auto endTime   = std::chrono::high_resolution_clock::now();
    auto dt        = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    auto tmo       = std::chrono::duration<int, std::micro>(tmoS * 1000000).count();
    totSize   = 0;
    startTime = std::chrono::high_resolution_clock::now();
    do
    {
      fileWriter.writeEvent(data_d, bufSize, ts);
      totSize += bufSize;

      endTime = std::chrono::high_resolution_clock::now();
      dt      = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    } while (dt < tmo);  // Run for awhile to see several points in grafana
    chkError(cudaStreamSynchronize(stream));
    fileWriter.close();

    endTime = std::chrono::high_resolution_clock::now();
    dt      = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

    return dt;
}


int main(int argc, char* argv[])
{
    [[maybe_unused]] size_t minRecSz = 8 * MB;  // 1 * kB
    size_t maxRecSz = 8 * MB;  // 8 * GB
    unsigned repeat = 1;
    unsigned durationS = 30;
    [[maybe_unused]] unsigned count = 1;
    std::string base("/home/claus/test/");
    unsigned gpuId = 0;
    unsigned verbose = 0;

    int c;
    while((c = getopt(argc, argv, "B:n:r:m:b:t:g:v")) != EOF)
    {
        switch(c)
        {
          case 'B':  base      = optarg;             break;
          case 'n':  count     = std::stoi(optarg);  break;
          case 'r':  repeat    = std::stoi(optarg);  break;
          case 't':  durationS = std::stoi(optarg);  break;
          case 'm':  minRecSz  = std::stoi(optarg);  break;
          case 'b':  maxRecSz  = std::stoi(optarg);  break;
          case 'g':  gpuId     = std::stoi(optarg);  break;
          case 'v':  ++verbose;                      break;
          default:
            printf("%s "
                   "[-B <file spec base>] "
                   "[-n <thread count>] "
                   "[-r <repeat count>]"
                   "[-t <duration (S)>]"
                   "[-m <min record size (B)>]"
                   "[-b <max record size (B)>]"
                   "[-g <GPU Id]"
                   "[-v]\n", argv[0]);
            return 1;
        }
    }

    switch (verbose) {
        case 0:  logging::init("tst", LOG_INFO);   break;
        default: logging::init("tst", LOG_DEBUG);  break;
    }

    chkError(cudaSetDevice(gpuId));     // Not obviously needed...

    //size_t totSize = repeat * maxRecSz;      // Total bytes per file
    //std::vector<uint64_t> data = GenerateData(count * totSize);
    std::vector<uint64_t> data(maxRecSz/sizeof(uint64_t)); // = GenerateData(maxRecSz);
    void* data_d;
    chkError(cudaMalloc(&data_d,              data.size() * sizeof(data[0])));
    chkError(cudaMemcpy( data_d, data.data(), data.size() * sizeof(data[0]), cudaMemcpyHostToDevice));

    size_t recSz = 32 * MB;   // Max pinned memory size
    size_t bufSz = data.size() * sizeof(data[0]);
    for (unsigned i = 0; i < repeat; ++i)
    {
      size_t totSize;
      auto dt = option_1(data_d, base + "1_gpu.bin", recSz, bufSz, durationS, totSize);

      std::cout << "FileWriter(" << recSz/MB << "MB), "
                << durationS << "s @ "
                << (bufSz > 1000 ? bufSz / kB : bufSz) << (bufSz > 1000 ? " kB: " : " B: ")
                << dt << " us, "
                << double(totSize) / double(dt) << " MB/s"
                << std::endl;
      usleep(5000000);                 // Wait 5 s so we see a notch in grafana
    }

    recSz = 16 * MB;          // Half the max pinned memory size to allow ping ponging
    for (unsigned i = 0; i < repeat; ++i)
    {
      size_t totSize;
      auto dt = option_2(data_d, base + "2_gpu.bin", recSz, bufSz, durationS, totSize);

      std::cout << "FileWriterAsync(" << recSz/MB << "MB), "
                << durationS << "s @ "
                << (bufSz > 1000 ? bufSz / kB : bufSz) << (bufSz > 1000 ? " kB: " : " B: ")
                << dt << " us, "
                << double(totSize) / double(dt) << " MB/s"
                << std::endl;
      usleep(5000000);                 // Wait 5 s so we see a notch in grafana
    }

    return 0;
}
