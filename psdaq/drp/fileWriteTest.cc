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
#include "psalg/utils/SysLog.hh"

#include "FileWriter.hh"

using logging = psalg::SysLog;

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

long long option_1(const std::vector<uint64_t>& data,
                   const std::string& filename,
                   size_t bytes,
                   size_t totSize)
{
  //{ // Warm up
  //  auto myfile = std::fstream(filename.c_str(), std::ios::out | std::ios::binary);
  //  myfile.write((char*)&data[0], bytes < 1 * MB ? bytes : 1 * MB);
  //  myfile.close();
    unlink(filename.c_str());           // Delete file
  //}
    size_t amount = 0;
    auto startTime = std::chrono::high_resolution_clock::now();
    auto myfile = std::fstream(filename.c_str(), std::ios::out | std::ios::binary);
    while (amount < totSize)
    {
      myfile.write((char*)&data[amount >> 3], bytes);
      amount += bytes;
    }
    myfile.close();
    auto endTime = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
}

long long option_2(const std::vector<uint64_t>& data,
                   const std::string& filename,
                   size_t bytes,
                   size_t totSize)
{
  //{ // Warm up
  //  FILE* file = fopen(filename.c_str(), "wb");
  //  if (!file) perror("fopen failed");
  //  fwrite(&data[0], 1, bytes < 1 * MB ? bytes : 1 * MB, file);
  //  fclose(file);
    unlink(filename.c_str());           // Delete file
  //}
    size_t amount = 0;
    auto startTime = std::chrono::high_resolution_clock::now();
    FILE* file = fopen(filename.c_str(), "wb");
    while (amount < totSize)
    {
      fwrite(&data[amount >> 3], 1, bytes, file);
      amount += bytes;
    }
    fclose(file);
    auto endTime = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
}

long long option_3(const std::vector<uint64_t>& data,
                   const std::string& filename,
                   size_t bytes,
                   size_t totSize)
{
    auto sync = std::ios_base::sync_with_stdio(false);

  //{ // Warm up
  //  auto myfile = std::fstream(filename.c_str(), std::ios::out | std::ios::binary);
  //  myfile.write((char*)&data[0], bytes < 1 * MB ? bytes : 1 * MB);
  //  myfile.close();
    unlink(filename.c_str());           // Delete file
  //}
    size_t amount = 0;
    auto startTime = std::chrono::high_resolution_clock::now();
    auto myfile = std::fstream(filename.c_str(), std::ios::out | std::ios::binary);
    while (amount < totSize)
    {
      myfile.write((char*)&data[amount >> 3], bytes);
      amount += bytes;
    }
    myfile.close();
    auto endTime = std::chrono::high_resolution_clock::now();

    std::ios_base::sync_with_stdio(sync);

    return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
}

long long option_4(const std::vector<uint64_t>& data,
                   const std::string& filename,
                   size_t bytes,
                   size_t totSize)
{
  //{ // Warm up
  //  int fd = ::open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP);
  //  if (fd == -1) perror("open failed");
  //  write(fd, &data[0], bytes < 1 * MB ? bytes : 1 * MB);
  //  ::close(fd);
    unlink(filename.c_str());           // Delete file
  //}
    size_t amount = 0;
    auto startTime = std::chrono::high_resolution_clock::now();
    int fd = ::open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP);
    while (amount < totSize)
    {
      auto sz = write(fd, &data[amount >> 3], bytes);
      while (size_t(sz) != bytes)
      {
        if (sz < 0)
        {
          perror("write failed");
          break;
        }
        bytes -= sz;
        sz = write(fd, (uint8_t*)&data[amount >> 3] + sz, bytes);
      }
      amount += bytes;
    }
    ::close(fd);
    auto endTime = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
}

long long option_5(const std::vector<uint64_t>& data,
                   const std::string& filename,
                   size_t recSize,
                   size_t bufSize,
                   unsigned tmoS,
                   size_t& totSize)
{
    const bool dio = true;              // Use Direct IO
    Drp::BufferedFileWriterMT fileWriter(recSize, dio); //8388688);
    //Drp::BufferedFileWriter fileWriter(recSize); //8388688);
    XtcData::TimeStamp ts(0, 0);

    unlink(filename.c_str());           // Delete file

    if (fileWriter.open(filename) != 0)  throw "open failed";
    fileWriter.writeEvent(data.data(), bufSize, ts); // Warm up
    fileWriter.flush();                              // Warm up
    auto startTime = std::chrono::high_resolution_clock::now();
    auto endTime   = std::chrono::high_resolution_clock::now();
    auto dt        = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    auto tmo       = std::chrono::duration<int, std::micro>(tmoS * 1000000).count();
    totSize   = 0;
    startTime = std::chrono::high_resolution_clock::now();
    do
    {
      fileWriter.writeEvent(data.data(), bufSize, ts);
      totSize += bufSize;

      endTime = std::chrono::high_resolution_clock::now();
      dt      = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    } while (dt < tmo);  // Run for awhile to see several points in grafana
    fileWriter.close();

    endTime = std::chrono::high_resolution_clock::now();
    dt      = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

    return dt;
}

long long option_6(const std::vector<uint64_t>& data,
                   const std::string& filename,
                   size_t bytes,
                   size_t totSize,
                   unsigned count)
{
    XtcData::TimeStamp ts(0, 0);        // No pauses between writes, so irrelevant
    std::vector<std::string> filenames;
    std::vector< std::unique_ptr<Drp::BufferedFileWriterMT> > fileWriters;
    char buf[80];

    for (auto i = 0u; i < count; ++i)
    {
      fileWriters.push_back(std::make_unique<Drp::BufferedFileWriterMT>(8388688));

      snprintf(buf, sizeof(buf), filename.c_str(), i);
      filenames.emplace_back(buf);

      unlink(filenames[i].c_str());     // Delete file
    }

    size_t amount = 0;
    unsigned j = 0;
    auto startTime = std::chrono::high_resolution_clock::now();
    for (auto i = 0u; i < fileWriters.size(); ++i)
      if (fileWriters[i]->open(filenames[i]) != 0)  throw "open failed";
    while (amount < totSize)
    {
      fileWriters[j]->writeEvent((uint8_t*)&data[amount >> 3] + j * bytes, bytes, ts);
      amount += bytes;
      j = (j + 1) % fileWriters.size();
    }
    for (auto i = 0u; i < fileWriters.size(); ++i)
      fileWriters[i]->close();
    auto endTime = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
}

#include <sstream>
#include <iomanip>      // std::setfill, std::setw

long long option_7(const std::vector<uint64_t>& data,
                   const std::string& filename,
                   size_t bytes,
                   size_t totSize,
                   unsigned count)
{
    // For this test, need to delete any previous file because the flags are set
    // so that it can't be overwritten, so this code is duplicated here
    // We don't do this in FileWriter because attempting to overwrite a data file
    // is, and should be, an error
    auto dot = filename.find_last_of(".");
    if (dot == std::string::npos)  logging::debug("No '.' in filename %s\n", filename.c_str());
    auto base(filename.substr(0, dot));
    auto ext(filename.substr(dot));

    for (auto i = 0u; i < count; ++i)
    {
      std::ostringstream ss;
      ss << base << "-i" << std::setfill('0') << std::setw(2) << i << ext;
      auto rc = unlink(ss.str().c_str());     // Delete file
      logging::debug("unlink %s: %d\n", ss.str().c_str(), rc);
      ss.seekp(0).clear();
    }

    XtcData::TimeStamp ts(0, 0);        // No pauses between writes, so irrelevant
    Drp::BufferedMultiFileWriterMT fileWriters(8388688, count);
    size_t amount = 0;
    unsigned j = 0;
    auto startTime = std::chrono::high_resolution_clock::now();
    if (fileWriters.open(filename) != 0)  { logging::debug("open failed\n"); return -1; }
    while (amount < totSize)
    {
      fileWriters.writeEvent((uint8_t*)&data[amount >> 3] + j * bytes, bytes, ts);
      amount += bytes;
      j = (j + 1) % count;
    }
    fileWriters.close();
    auto endTime = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
}

static inline ssize_t _write(int fd, const void* buffer, size_t count)
{
    auto sz = write(fd, buffer, count);
    while (size_t(sz) != count) {
        if (sz < 0) {
            // %m will be replaced by the string strerror(errno)
            logging::error("write error: %m");
            return sz;
        }
        count -= sz;
        sz = write(fd, (uint8_t*)buffer + sz, count);
    }
    return sz;
}

long long option_8(const std::vector<uint64_t>& data,
                   const std::string& filename,
                   size_t bytes,
                   size_t totSize)
{
    unlink(filename.c_str());

    auto fd = ::open(filename.c_str(), O_DIRECT | O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP);
    if (fd == -1)
    {
      // %m will be replaced by the string strerror(errno)
      logging::error("Error creating file %s: %m", filename.c_str());
    }

    auto startTime = std::chrono::high_resolution_clock::now();
    size_t amount = 0;
    while (amount < totSize)
    {
      _write(fd, data.data(), bytes);
      amount += bytes;
    }

    ::close(fd);
    auto endTime = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
}


int main(int argc, char* argv[])
{
//    size_t minRecSz = 8 * MB;  // 1 * kB
    size_t maxRecSz = 8 * MB;  // 8 * GB
    unsigned repeat = 1;
    unsigned durationS = 30;
//    unsigned count = 1;
    std::string base("/u1/claus/opt");
    unsigned verbose = 0;

    int c;
    while((c = getopt(argc, argv, "B:n:r:m:b:t:v")) != EOF)
    {
        switch(c)
        {
          case 'B':  base      = optarg;                  break;
//          case 'n':  count     = std::stoi(optarg);       break;
          case 'r':  repeat    = std::stoi(optarg);       break;
          case 't':  durationS = std::stoi(optarg);       break;
//          case 'm':  minRecSz  = std::stoi(optarg) * MB;  break;
          case 'b':  maxRecSz  = std::stoi(optarg) * MB;  break;
          case 'v':  ++verbose;                           break;
          default:
            printf("%s "
                   "[-B <file spec base>] "
                   "[-n <thread count>] "
                   "[-r <repeat count>]"
                   "[-t <duration (S)>]"
                   "[-m <min record size (MB)>]"
                   "[-b <max record size (MB)>]"
                   "[-v]\n", argv[0]);
            return 1;
        }
    }

    switch (verbose) {
        case 0:  logging::init("tst", LOG_INFO);   break;
        default: logging::init("tst", LOG_DEBUG);  break;
    }

    //size_t totSize = repeat * maxRecSz;      // Total bytes per file
    //std::vector<uint64_t> data = GenerateData(count * totSize);
    std::vector<uint64_t> data(maxRecSz/8); // = GenerateData(maxRecSz);

    //for (size_t size = 1 * kB; size <= 8 * MB; size *= 2) { auto dt = option_1(data, base + "1.bin", size, totSize);             std::cout << "option 1, "                      << size / kB << " kB: " << dt << " us, " << double(totSize) / double(dt) << " MB/s" << std::endl; }  std::cout << "\n\n";
    //for (size_t size = 1 * kB; size <= 8 * MB; size *= 2) { auto dt = option_2(data, base + "2.bin", size, totSize);             std::cout << "option 2, "                      << size / kB << " kB: " << dt << " us, " << double(totSize) / double(dt) << " MB/s" << std::endl; }  std::cout << "\n\n";
    //for (size_t size = 1 * kB; size <= 8 * MB; size *= 2) { auto dt = option_3(data, base + "3.bin", size, totSize);             std::cout << "option 3, "                      << size / kB << " kB: " << dt << " us, " << double(totSize) / double(dt) << " MB/s" << std::endl; }  std::cout << "\n\n";
    //for (size_t size = 1 * kB; size <= 8 * MB; size *= 2) { auto dt = option_4(data, base + "4.bin", size, totSize);             std::cout << "option 4, "                      << size / kB << " kB: " << dt << " us, " << double(totSize) / double(dt) << " MB/s" << std::endl; }  std::cout << "\n\n";
    //for (size_t size = 1 * kB; size <= 8 * MB; size *= 2) { auto dt = option_5(data, base + "5.bin", size, totSize);             std::cout << "option 5, "                      << size / kB << " kB: " << dt << " us, " << double(totSize) / double(dt) << " MB/s" << std::endl; }  std::cout << "\n\n";
    //for (size_t size = 1 * kB; size <= 8 * MB; size *= 2) { auto dt = option_6(data, base + "61_%u.bin", size, totSize, 1);      std::cout << "option 6(1), " << 1     << " @ " << size / kB << " kB: " << dt << " us, " << double(totSize) / double(dt) << " MB/s" << std::endl; }  std::cout << "\n\n";
    //for (size_t size = 1 * kB; size <= 8 * MB; size *= 2) { auto dt = option_6(data, base + "6N_%u.bin", size, totSize, count);  std::cout << "option 6(N), " << count << " @ " << size / kB << " kB: " << dt << " us, " << double(totSize) / double(dt) << " MB/s" << std::endl; }  std::cout << "\n\n";
    //for (size_t size = 1 * kB; size <= 8 * MB; size *= 2) { auto dt = option_7(data, base + "7N.bin", size, totSize, count);     std::cout << "option 7(N), " << count << " @ " << size / kB << " kB: " << dt << " us, " << double(totSize) / double(dt) << " MB/s" << std::endl; }  std::cout << "\n\n";

    //for (size_t size = 1 * kB; size <= 8 * MB; size *= 2) { auto dt = option_7(data, base + "701.bin", size, totSize,  1);     std::cout << "option 7(1), "  <<  1 << " @ " << size / kB << " kB: " << dt << " us, " << double(totSize) / double(dt) << " MB/s" << std::endl; }  std::cout << "\n\n";
    //for (size_t size = 1 * kB; size <= 8 * MB; size *= 2) { auto dt = option_7(data, base + "701.bin", size, totSize,  1);     std::cout << "option 7(1), "  <<  1 << " @ " << size / kB << " kB: " << dt << " us, " << double(totSize) / double(dt) << " MB/s" << std::endl; }  std::cout << "\n\n";
    //for (size_t size = 1 * kB; size <= 8 * MB; size *= 2) { auto dt = option_7(data, base + "701.bin", size, totSize,  1);     std::cout << "option 7(1), "  <<  1 << " @ " << size / kB << " kB: " << dt << " us, " << double(totSize) / double(dt) << " MB/s" << std::endl; }  std::cout << "\n\n";
    //
    //for (size_t size = 1 * kB; size <= 8 * MB; size *= 2) { auto dt = option_7(data, base + "702.bin", size, totSize,  2);     std::cout << "option 7(2), "  <<  2 << " @ " << size / kB << " kB: " << dt << " us, " << double(totSize) / double(dt) << " MB/s" << std::endl; }  std::cout << "\n\n";
    //for (size_t size = 1 * kB; size <= 8 * MB; size *= 2) { auto dt = option_7(data, base + "702.bin", size, totSize,  2);     std::cout << "option 7(2), "  <<  2 << " @ " << size / kB << " kB: " << dt << " us, " << double(totSize) / double(dt) << " MB/s" << std::endl; }  std::cout << "\n\n";
    //for (size_t size = 1 * kB; size <= 8 * MB; size *= 2) { auto dt = option_7(data, base + "702.bin", size, totSize,  2);     std::cout << "option 7(2), "  <<  2 << " @ " << size / kB << " kB: " << dt << " us, " << double(totSize) / double(dt) << " MB/s" << std::endl; }  std::cout << "\n\n";
    //
    //for (size_t size = 1 * kB; size <= 8 * MB; size *= 2) { auto dt = option_7(data, base + "704.bin", size, totSize,  4);     std::cout << "option 7(4), "  <<  4 << " @ " << size / kB << " kB: " << dt << " us, " << double(totSize) / double(dt) << " MB/s" << std::endl; }  std::cout << "\n\n";
    //for (size_t size = 1 * kB; size <= 8 * MB; size *= 2) { auto dt = option_7(data, base + "704.bin", size, totSize,  4);     std::cout << "option 7(4), "  <<  4 << " @ " << size / kB << " kB: " << dt << " us, " << double(totSize) / double(dt) << " MB/s" << std::endl; }  std::cout << "\n\n";
    //for (size_t size = 1 * kB; size <= 8 * MB; size *= 2) { auto dt = option_7(data, base + "704.bin", size, totSize,  4);     std::cout << "option 7(4), "  <<  4 << " @ " << size / kB << " kB: " << dt << " us, " << double(totSize) / double(dt) << " MB/s" << std::endl; }  std::cout << "\n\n";
    //
    //for (size_t size = 1 * kB; size <= 8 * MB; size *= 2) { auto dt = option_7(data, base + "708.bin", size, totSize,  8);     std::cout << "option 7(8), "  <<  8 << " @ " << size / kB << " kB: " << dt << " us, " << double(totSize) / double(dt) << " MB/s" << std::endl; }  std::cout << "\n\n";
    //for (size_t size = 1 * kB; size <= 8 * MB; size *= 2) { auto dt = option_7(data, base + "708.bin", size, totSize,  8);     std::cout << "option 7(8), "  <<  8 << " @ " << size / kB << " kB: " << dt << " us, " << double(totSize) / double(dt) << " MB/s" << std::endl; }  std::cout << "\n\n";
    //for (size_t size = 1 * kB; size <= 8 * MB; size *= 2) { auto dt = option_7(data, base + "708.bin", size, totSize,  8);     std::cout << "option 7(8), "  <<  8 << " @ " << size / kB << " kB: " << dt << " us, " << double(totSize) / double(dt) << " MB/s" << std::endl; }  std::cout << "\n\n";
    //
    //for (size_t size = 1 * kB; size <= 8 * MB; size *= 2) { auto dt = option_7(data, base + "716.bin", size, totSize, 16);     std::cout << "option 7(16), " << 16 << " @ " << size / kB << " kB: " << dt << " us, " << double(totSize) / double(dt) << " MB/s" << std::endl; }  std::cout << "\n\n";
    //for (size_t size = 1 * kB; size <= 8 * MB; size *= 2) { auto dt = option_7(data, base + "716.bin", size, totSize, 16);     std::cout << "option 7(16), " << 16 << " @ " << size / kB << " kB: " << dt << " us, " << double(totSize) / double(dt) << " MB/s" << std::endl; }  std::cout << "\n\n";
    //for (size_t size = 1 * kB; size <= 8 * MB; size *= 2) { auto dt = option_7(data, base + "716.bin", size, totSize, 16);     std::cout << "option 7(16), " << 16 << " @ " << size / kB << " kB: " << dt << " us, " << double(totSize) / double(dt) << " MB/s" << std::endl; }  std::cout << "\n\n";
    //
    //for (size_t size = 1 * kB; size <= 8 * MB; size *= 2) { auto dt = option_7(data, base + "732.bin", size, totSize, 32);     std::cout << "option 7(32), " << 32 << " @ " << size / kB << " kB: " << dt << " us, " << double(totSize) / double(dt) << " MB/s" << std::endl; }  std::cout << "\n\n";
    //for (size_t size = 1 * kB; size <= 8 * MB; size *= 2) { auto dt = option_7(data, base + "732.bin", size, totSize, 32);     std::cout << "option 7(32), " << 32 << " @ " << size / kB << " kB: " << dt << " us, " << double(totSize) / double(dt) << " MB/s" << std::endl; }  std::cout << "\n\n";
    //for (size_t size = 1 * kB; size <= 8 * MB; size *= 2) { auto dt = option_7(data, base + "732.bin", size, totSize, 32);     std::cout << "option 7(32), " << 32 << " @ " << size / kB << " kB: " << dt << " us, " << double(totSize) / double(dt) << " MB/s" << std::endl; }  std::cout << "\n\n";
    //
    //for (size_t size = 1 * kB; size <= 8 * MB; size *= 2) { auto dt = option_7(data, base + "764.bin", size, totSize, 64);     std::cout << "option 7(64), " << 64 << " @ " << size / kB << " kB: " << dt << " us, " << double(totSize) / double(dt) << " MB/s" << std::endl; }  std::cout << "\n\n";
    //for (size_t size = 1 * kB; size <= 8 * MB; size *= 2) { auto dt = option_7(data, base + "764.bin", size, totSize, 64);     std::cout << "option 7(64), " << 64 << " @ " << size / kB << " kB: " << dt << " us, " << double(totSize) / double(dt) << " MB/s" << std::endl; }  std::cout << "\n\n";
    //for (size_t size = 1 * kB; size <= 8 * MB; size *= 2) { auto dt = option_7(data, base + "764.bin", size, totSize, 64);     std::cout << "option 7(64), " << 64 << " @ " << size / kB << " kB: " << dt << " us, " << double(totSize) / double(dt) << " MB/s" << std::endl; }  std::cout << "\n\n";

    //for (size_t size = 1 * kB; size <= 8 * MB; size *= 2) { auto dt = option_5(data, base + "5.bin", totSize, size, totSize);  std::cout << "option 5, " << repeat << " @ " << size / kB << " kB: " << dt / count << " us, " << double(totSize) / double(dt) << " MB/s" << std::endl; }  std::cout << "\n\n";
    //for (size_t recSz = minRecSz; recSz <= maxRecSz; recSz *= 2)
    //{
    //  size_t totSize = recSz * repeat;
    //  for (size_t size = 1 * kB; size <= 8 * MB; size *= 2)
    //  {
    //    auto dt = option_5(data, base + "5.bin", recSz, size, totSize);
    //    std::cout << "BufferedFileWriterMT(" << recSz/MB << "MB), "
    //              << repeat << " @ " << size / kB << " kB: "
    //              << dt / count << " us, "
    //              << double(totSize) / double(dt) << " MB/s"
    //              << std::endl;
    //  }
    //  std::cout << "\n\n";
    //}

    //for (size_t recSz = minRecSz; recSz <= maxRecSz; recSz *= 2)  // Revisit: Adjust BufferedFileWriterMT for this
    size_t recSz = 512 * MB;   // Revisit: Same as FileWriter.FIFO_MIN_SIZE
    //size_t recSz = 8388688;
    //size_t bufSz = 16384;
    size_t size  = recSz;
    //printf("repeat %u, recSz 0x%zx, dataSize %zu\n", repeat, recSz, data.size() * sizeof(data[0]));
    for (unsigned i = 0; i < repeat; ++i)
    //for (size_t size = bufSz; size > 1; size >>= 1)
    {
      //printf("size %zu\n", size);
      size_t totSize;
      auto dt = option_5(data, base + "5.bin", recSz, size, durationS, totSize);

      std::cout << "BufferedFileWriterMT(" << recSz/MB << "MB), "
                << durationS << "S @ "
                << (size > 1000 ? size / kB : size) << (size > 1000 ? " kB: " : " B: ")
                << dt << " us, "
                << double(totSize) / double(dt) << " MB/s"
                << std::endl;
      usleep(5000000);                 // Wait 5 s so we see a notch in grafana
      //std::cout << "\n\n";
    }

    //for (size_t size = minRecSz; size <= maxRecSz; size *= 2) { auto dt = option_8(data, base + "8.bin", size, totSize);         std::cout << "option 8, " << repeat << " @ " << size / kB << " kB: " << dt / count << " us, " << double(totSize) / double(dt) / double(count) << " MB/s" << std::endl; }  std::cout << "\n\n";
    //size_t size = maxRecSz; //8 * MB;
    //auto dt = option_8(data, base + "801.bin", size, totSize);
    //std::cout << "option 8, " << repeat << " @ " << size / kB << " kB: " << dt / count << " us, " << double(totSize) / double(dt) / double(count) << " MB/s" << std::endl;
    //printf("size %zu, totSize %zu, dt %lld, rate %f\n", size, totSize, dt, double(totSize)/double(dt));

    return 0;
}
