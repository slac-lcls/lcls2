#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kDgramHeaderSize = 24;
constexpr std::size_t kXtcHeaderSize = 12;
constexpr std::size_t kDefaultMaxDgramsPerFile = 2;
constexpr std::size_t kMaxDgramSize = 0x04000000;
constexpr std::size_t kMaxChildTypes = 64;
constexpr const char* kDefaultXtcDir = "/sdf/data/lcls/ds/mfx/mfx101344525/xtc";

enum SkeletonStatus : int32_t {
  kOk = 0,
  kTooSmall = 1,
  kSizeMismatch = 2,
  kInvalidRootExtent = 3,
  kRootOverflow = 4,
  kInvalidChildExtent = 5,
  kChildOverflow = 6,
};

struct GpuDgramSummary
{
  uint64_t timestamp;
  uint64_t file_offset;
  uint64_t total_bytes;
  uint32_t env;
  uint32_t xtc_src;
  uint32_t xtc_extent;
  uint16_t xtc_damage;
  uint16_t xtc_type_id;
  uint16_t xtc_version;
  uint16_t service;
  uint16_t transition_type;
  uint16_t readout_groups;
  uint32_t child_count;
  uint32_t child_stored;
  int32_t  status;
  uint16_t child_type_ids[kMaxChildTypes];
};

struct HostDgramRecord
{
  std::string file_path;
  uint64_t file_offset;
  std::vector<uint8_t> bytes;
};

struct Options
{
  int device = 0;
  std::size_t max_dgrams_per_file = kDefaultMaxDgramsPerFile;
  std::vector<std::string> files;
};

inline void checkCuda(cudaError_t err, const char* what)
{
  if (err != cudaSuccess) {
    std::ostringstream os;
    os << what << ": " << cudaGetErrorString(err);
    throw std::runtime_error(os.str());
  }
}

__host__ __device__ inline uint16_t load_u16(const uint8_t* p)
{
  return static_cast<uint16_t>(p[0]) |
         static_cast<uint16_t>(static_cast<uint16_t>(p[1]) << 8);
}

__host__ __device__ inline uint32_t load_u32(const uint8_t* p)
{
  return static_cast<uint32_t>(p[0]) |
         (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

__global__ void materializeGpudgramSkeleton(
    const uint8_t* data,
    std::size_t nbytes,
    uint64_t file_offset,
    GpuDgramSummary* out)
{
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  GpuDgramSummary summary = {};
  summary.file_offset = file_offset;
  summary.total_bytes = nbytes;
  for (std::size_t i = 0; i < kMaxChildTypes; ++i) {
    summary.child_type_ids[i] = 0xffff;
  }

  if (nbytes < kDgramHeaderSize) {
    summary.status = kTooSmall;
    *out = summary;
    return;
  }

  const uint32_t nsec = load_u32(data + 0);
  const uint32_t sec = load_u32(data + 4);
  const uint32_t env = load_u32(data + 8);
  const uint32_t xtc_src = load_u32(data + 12);
  const uint16_t xtc_damage = load_u16(data + 16);
  const uint16_t xtc_contains = load_u16(data + 18);
  const uint32_t xtc_extent = load_u32(data + 20);

  summary.timestamp = (static_cast<uint64_t>(sec) << 32) | nsec;
  summary.env = env;
  summary.xtc_src = xtc_src;
  summary.xtc_damage = xtc_damage;
  summary.xtc_extent = xtc_extent;
  summary.xtc_type_id = xtc_contains & 0x0fff;
  summary.xtc_version = (xtc_contains >> 12) & 0x000f;

  const uint32_t control = (env >> 24) & 0xff;
  summary.service = control & 0x0f;
  summary.transition_type = (control >> 4) & 0x03;
  summary.readout_groups = env & 0xffff;

  if (xtc_extent < kXtcHeaderSize) {
    summary.status = kInvalidRootExtent;
    *out = summary;
    return;
  }

  const std::size_t computed_total = kDgramHeaderSize + (xtc_extent - kXtcHeaderSize);
  if (computed_total != nbytes) {
    summary.status = kSizeMismatch;
    *out = summary;
    return;
  }

  const std::size_t root_begin = 12;
  const std::size_t root_end = root_begin + xtc_extent;
  if (root_end > nbytes) {
    summary.status = kRootOverflow;
    *out = summary;
    return;
  }

  if (summary.xtc_type_id == 0) {
    std::size_t child_offset = kDgramHeaderSize;
    while (child_offset < root_end) {
      if (child_offset + kXtcHeaderSize > nbytes) {
        summary.status = kChildOverflow;
        *out = summary;
        return;
      }

      const uint16_t child_contains = load_u16(data + child_offset + 6);
      const uint32_t child_extent = load_u32(data + child_offset + 8);
      if (child_extent < kXtcHeaderSize) {
        summary.status = kInvalidChildExtent;
        *out = summary;
        return;
      }
      if (child_offset + child_extent > root_end || child_offset + child_extent > nbytes) {
        summary.status = kChildOverflow;
        *out = summary;
        return;
      }

      if (summary.child_stored < kMaxChildTypes) {
        summary.child_type_ids[summary.child_stored] = child_contains & 0x0fff;
        ++summary.child_stored;
      }
      ++summary.child_count;
      child_offset += child_extent;
    }
  }

  summary.status = kOk;
  *out = summary;
}

std::string serviceName(uint16_t service)
{
  switch (service) {
    case 0: return "ClearReadout";
    case 1: return "Reset";
    case 2: return "Configure";
    case 3: return "Unconfigure";
    case 4: return "BeginRun";
    case 5: return "EndRun";
    case 6: return "BeginStep";
    case 7: return "EndStep";
    case 8: return "Enable";
    case 9: return "Disable";
    case 10: return "SlowUpdate";
    case 11: return "Unused_11";
    case 12: return "L1Accept";
    default: {
      std::ostringstream os;
      os << "Transition(" << service << ")";
      return os.str();
    }
  }
}

std::string typeName(uint16_t type_id)
{
  switch (type_id) {
    case 0: return "Parent";
    case 1: return "ShapesData";
    case 2: return "Shapes";
    case 3: return "Data";
    case 4: return "Names";
    default: {
      std::ostringstream os;
      os << "TypeId(" << type_id << ")";
      return os.str();
    }
  }
}

std::string statusName(int32_t status)
{
  switch (status) {
    case kOk: return "ok";
    case kTooSmall: return "too_small";
    case kSizeMismatch: return "size_mismatch";
    case kInvalidRootExtent: return "invalid_root_extent";
    case kRootOverflow: return "root_overflow";
    case kInvalidChildExtent: return "invalid_child_extent";
    case kChildOverflow: return "child_overflow";
    default: {
      std::ostringstream os;
      os << "status(" << status << ")";
      return os.str();
    }
  }
}

std::string childTypeSummary(const GpuDgramSummary& summary)
{
  std::ostringstream os;
  os << "[";
  for (uint32_t i = 0; i < summary.child_stored; ++i) {
    if (i) {
      os << ", ";
    }
    os << typeName(summary.child_type_ids[i]);
  }
  if (summary.child_count > summary.child_stored) {
    if (summary.child_stored) {
      os << ", ";
    }
    os << "...";
  }
  os << "]";
  return os.str();
}

std::vector<std::string> defaultFiles()
{
  std::vector<std::string> files;
  for (int stream = 7; stream <= 11; ++stream) {
    std::ostringstream os;
    os << kDefaultXtcDir
       << "/mfx101344525-r0125-s"
       << std::setfill('0') << std::setw(3) << stream
       << "-c000.xtc2";
    files.push_back(os.str());
  }
  return files;
}

void printUsage(const char* argv0)
{
  std::cout
      << "Usage: " << argv0 << " [--device N] [--max-dgrams-per-file N] [--files file1 file2 ...]\n"
      << "Defaults to mfx101344525:r0125 Jungfrau streams s007-s011 c000.\n";
}

Options parseArgs(int argc, char** argv)
{
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--device") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--device requires an argument");
      }
      options.device = std::stoi(argv[++i]);
    } else if (arg == "--max-dgrams-per-file") {
      if (i + 1 >= argc) {
        throw std::runtime_error("--max-dgrams-per-file requires an argument");
      }
      options.max_dgrams_per_file = static_cast<std::size_t>(std::stoul(argv[++i]));
    } else if (arg == "--files") {
      while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
        options.files.emplace_back(argv[++i]);
      }
    } else if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (options.files.empty()) {
    options.files = defaultFiles();
  }
  return options;
}

bool readNextDgram(
    std::ifstream& input,
    const std::string& file_path,
    uint64_t& next_offset,
    HostDgramRecord& record)
{
  std::array<uint8_t, kDgramHeaderSize> header = {};
  input.read(reinterpret_cast<char*>(header.data()), header.size());
  const std::streamsize got = input.gcount();
  if (got == 0) {
    return false;
  }
  if (got != static_cast<std::streamsize>(header.size())) {
    throw std::runtime_error("Incomplete dgram header in " + file_path);
  }

  const uint32_t xtc_extent = load_u32(header.data() + 20);
  if (xtc_extent < kXtcHeaderSize) {
    throw std::runtime_error("Invalid xtc extent in " + file_path);
  }

  const std::size_t total_bytes = kDgramHeaderSize + (xtc_extent - kXtcHeaderSize);
  if (total_bytes > kMaxDgramSize) {
    throw std::runtime_error("Dgram exceeds max size in " + file_path);
  }

  record.file_path = file_path;
  record.file_offset = next_offset;
  record.bytes.assign(total_bytes, 0);
  std::copy(header.begin(), header.end(), record.bytes.begin());

  const std::size_t payload_bytes = total_bytes - header.size();
  input.read(
      reinterpret_cast<char*>(record.bytes.data() + header.size()),
      static_cast<std::streamsize>(payload_bytes));
  if (input.gcount() != static_cast<std::streamsize>(payload_bytes)) {
    throw std::runtime_error("Incomplete dgram payload in " + file_path);
  }

  next_offset += total_bytes;
  return true;
}

void printDeviceBanner(int device)
{
  cudaDeviceProp prop = {};
  checkCuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
  std::cout << "Using CUDA device " << device << " (" << prop.name << ")\n";
}

} // namespace

int main(int argc, char** argv)
{
  try {
    const Options options = parseArgs(argc, argv);

    int device_count = 0;
    checkCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < 1) {
      throw std::runtime_error("No CUDA devices detected");
    }
    if (options.device < 0 || options.device >= device_count) {
      throw std::runtime_error("Requested CUDA device is out of range");
    }
    checkCuda(cudaSetDevice(options.device), "cudaSetDevice");
    printDeviceBanner(options.device);

    std::cout << "GPUDgram skeleton runner\n";
    std::cout << "files=" << options.files.size()
              << " max_dgrams_per_file=" << options.max_dgrams_per_file
              << "\n";
    for (const auto& file : options.files) {
      std::cout << "  input=" << file << "\n";
    }

    uint8_t* d_bytes = nullptr;
    std::size_t d_capacity = 0;
    GpuDgramSummary* d_summary = nullptr;
    checkCuda(cudaMalloc(&d_summary, sizeof(GpuDgramSummary)), "cudaMalloc(d_summary)");

    for (const auto& file : options.files) {
      std::ifstream input(file, std::ios::binary);
      if (!input) {
        throw std::runtime_error("Failed to open " + file);
      }

      uint64_t next_offset = 0;
      std::size_t read_count = 0;
      HostDgramRecord record;
      while (read_count < options.max_dgrams_per_file &&
             readNextDgram(input, file, next_offset, record)) {
        if (record.bytes.size() > d_capacity) {
          if (d_bytes != nullptr) {
            checkCuda(cudaFree(d_bytes), "cudaFree(d_bytes)");
          }
          d_capacity = record.bytes.size();
          checkCuda(cudaMalloc(&d_bytes, d_capacity), "cudaMalloc(d_bytes)");
        }

        checkCuda(
            cudaMemcpy(d_bytes, record.bytes.data(), record.bytes.size(), cudaMemcpyHostToDevice),
            "cudaMemcpy H2D");
        materializeGpudgramSkeleton<<<1, 1>>>(d_bytes, record.bytes.size(), record.file_offset, d_summary);
        checkCuda(cudaGetLastError(), "materializeGpudgramSkeleton launch");
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        GpuDgramSummary summary = {};
        checkCuda(
            cudaMemcpy(&summary, d_summary, sizeof(summary), cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H summary");

        const std::size_t slash = record.file_path.find_last_of('/');
        const std::string file_name = slash == std::string::npos
            ? record.file_path
            : record.file_path.substr(slash + 1);
        std::cout << file_name
                  << " offset=" << summary.file_offset
                  << " service=" << serviceName(summary.service)
                  << " ts=" << summary.timestamp
                  << " total_bytes=" << summary.total_bytes
                  << " xtc_type=" << typeName(summary.xtc_type_id)
                  << " xtc_extent=" << summary.xtc_extent
                  << " children=" << summary.child_count << " " << childTypeSummary(summary)
                  << " device_ptr=" << static_cast<const void*>(d_bytes)
                  << " status=" << statusName(summary.status)
                  << "\n";

        ++read_count;
      }
    }

    if (d_bytes != nullptr) {
      checkCuda(cudaFree(d_bytes), "cudaFree(d_bytes)");
    }
    checkCuda(cudaFree(d_summary), "cudaFree(d_summary)");
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "gpudgramSkeleton error: " << ex.what() << "\n";
    return 1;
  }
}
