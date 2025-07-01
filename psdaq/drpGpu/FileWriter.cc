#include "FileWriter.hh"

#include "psalg/utils/SysLog.hh"
#include "GpuAsyncLib.hh"

#include <fcntl.h>
#include <assert.h>
#include <unistd.h>

using logging = psalg::SysLog;
using namespace XtcData;
using namespace Drp;
using namespace Drp::Gpu;


/// For dealing with the errors in cuFile structs

static
int chkCuda(CUfileError_t status)
{
  if (IS_CUDA_ERR(status)) {
    const char *str;
    CUresult strResult = cuGetErrorString(status.cu_err, &str);
    if (strResult == CUDA_SUCCESS) {
      std::cout << "CUDA ERROR: " << str << std::endl;
      return -2;
    } else {
      return -3;
    }
  }
  return 0;
}

static
int chkCuFile(CUfileError_t status)
{
  if (IS_CUFILE_ERR(status.err)) {
    std::cout << "cuFile ERROR " << status.err << ": "
              << CUFILE_ERRSTR(status.err) << std::endl;
    if (status.err == 5011)
      chkCuda(status);
    return -1;
  }
  return chkCuda(status);
}


FileWriter::FileWriter(size_t bufferSize, bool dio, cudaStream_t& stream) :
  FileWriterBase   (),
  m_fd             (0),
  m_stream         (stream),
  m_count          (0),
  m_batch_starttime(0, 0),
  m_bufferSize     (bufferSize),
  m_buffer_d       (nullptr),
  m_fileOffset     (0),
  m_dio            (dio)
{
  if (chkCuFile(cuFileDriverOpen())) {
    logging::critical("Error opening cuFile driver");
    abort();
  }

  if (bufferSize) {
    chkError(cudaMalloc(&m_buffer_d, bufferSize));
  }
}

FileWriter::~FileWriter()
{
  close();

  if (m_buffer_d) {
    chkError(cudaFree(m_buffer_d));
  }

  if (chkCuFile(cuFileDriverClose())) {
    logging::critical("Error closinging cuFile driver");
    abort();
  }
}

void FileWriter::dumpProperties() const
{
  CUfileDrvProps_t devProps;
  if (chkCuFile(cuFileDriverGetProperties(&devProps))) {
    std::cout << "Cannot read cuFile capabilities." << std::endl;
    return;
  }

  std::cout << "cuFile major version:    " << devProps.nvfs.major_version      << std::endl
            << "cuFile minor version:    " << devProps.nvfs.minor_version      << std::endl
            << "cuFile poll thresh size: " << devProps.nvfs.poll_thresh_size   << std::endl
            << "cuFile max dir. IO size: " << devProps.nvfs.max_direct_io_size << std::endl
            << "cuFile dstatus flags:    " << devProps.nvfs.dstatusflags       << std::endl
            << "cuFile dcontrol flags:   " << devProps.nvfs.dcontrolflags      << std::endl;

  std::cout << "cuFile max device cache size:   " << devProps.max_device_cache_size      << std::endl
            << "cuFile per buffer cache size:   " << devProps.per_buffer_cache_size      << std::endl
            << "cuFile max pin. memory size:    " << devProps.max_device_pinned_mem_size << std::endl
            << "cuFile max batch io timeout ms: " << devProps.max_batch_io_timeout_msecs << std::endl
            << "cuFile max batch io:            " << devProps.max_batch_io_size          << std::endl;
}

int FileWriter::open(const std::string& fileName)
{
  if (m_fd > 0) {
    logging::warning("open() closed an already open file");
    close();
  }

  auto oFlags = O_WRONLY | O_CREAT | O_TRUNC;
  if (m_dio)  oFlags |= O_DIRECT;
  int rc = ::open(fileName.c_str(), oFlags, S_IRUSR | S_IRGRP);
  if (rc == -1) {
    // %m will be replaced by the string strerror(errno)
    logging::error("Error creating file %s: %m", fileName.c_str());
    return rc;
  }
  m_fd = rc;

  // Establish write lock on the file
  struct flock flk;
  flk.l_type   = F_WRLCK;
  flk.l_whence = SEEK_SET;
  flk.l_start  = 0;
  flk.l_len    = 0;

  do {
    rc = fcntl(m_fd, F_SETLKW, &flk);
  } while (rc<0 && errno==EINTR);
  if (rc<0) {
    // %m will be replaced by the string strerror(errno)
    logging::error("Error locking file %s: %m", fileName.c_str());
  } else {
    rc = 0;                           // return OK
  }

  CUfileDescr_t descr;
  memset(reinterpret_cast<void *>(&descr), 0, sizeof(CUfileDescr_t));

  descr.handle.fd = m_fd;
  descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  if ( (rc = chkCuFile(cuFileHandleRegister(&m_handle, &descr))) ) {
    logging::error("Failed to register file handle for fd %d", m_fd);
    close();
    return rc;
  }

  if (m_buffer_d) {
    if ( (rc = chkCuFile(cuFileBufRegister(m_buffer_d, m_bufferSize, 0))) ) {
      logging::error("Failed to register GPU buffer %p, size %zu with cuFile",
                     m_buffer_d, m_bufferSize);
      close();
      return rc;
    }
  }

  return rc;
}

int FileWriter::close()
{
  if (m_buffer_d) {
    if (chkCuFile(cuFileBufDeregister(m_buffer_d))) {
      logging::error("Failed to deregister GPU buffer at %p with cuFile", m_buffer_d);
    }
  }

  int rc = 0;
  if (m_fd > 0) {
    if (m_count > 0) {
      logging::debug("Flushing %zu bytes to fd %d", m_count, m_fd);
      m_writing += 2;
      _write(m_buffer_d, m_count);
      m_writing -= 2;
      m_count = 0;
      m_batch_starttime = XtcData::TimeStamp(0,0);
    }
    cuFileHandleDeregister(m_handle);
    logging::debug("Closing fd %d", m_fd);
    rc = ::close(m_fd);
    if (rc == -1) {
      // %m will be replaced by the string strerror(errno)
      logging::error("Error closing fd %d: %m", m_fd);
    }
    m_fd = 0;
  }
  return rc;
}

ssize_t FileWriter::_write(const void* devPtr, size_t size)
{
  printf("*** FileWriter::write: 1 ptr %p, sz %zu, fileOffset %zd\n", devPtr, size, m_fileOffset);
  ssize_t rc = cuFileWrite(m_handle, devPtr, size, m_fileOffset, 0);
  if (rc > 0) {
    m_fileOffset += rc;
  } else {
    if (IS_CUFILE_ERR(rc))
      logging::error("Write error: devPtr %p, size %zu: %s (%zd)", devPtr, size, CUFILE_ERRSTR(rc), rc);
    else
      logging::error("Write error: devPtr %p, size %zu: %m", devPtr, size);
  }
  printf("*** FileWriter::write: 2 ptr %p, sz %zu\n", devPtr, size);

  return rc;
}

void FileWriter::writeEvent(const void* devPtr, size_t size, const TimeStamp timestamp)
{
  printf("*** FileWriter::write: 1 ptr %p, sz %zu, t %u.%09u\n", devPtr, size, timestamp.seconds(), timestamp.nanoseconds());

  // triggered only when starting from scratch
  if (m_batch_starttime.value()==0) m_batch_starttime = timestamp;

  // rough calculation: ignore nanoseconds
  unsigned age_seconds = timestamp.seconds()-m_batch_starttime.seconds();
  // write out data if buffer full or batch is too old
  // can't be 1 second without a more precise age calculation, since
  // the seconds field could have "rolled over" since the last event
  printf("*** FileWriter::write: 2 bufSz %zu, cnt %zu, age %u\n", m_bufferSize, m_count, age_seconds);
  if ((size > (m_bufferSize - m_count)) || age_seconds>2) {
    printf("*** FileWriter::write: 3 buf %p, cnt %zu\n", m_buffer_d, m_count);
    m_writing += 1;
    auto rc = _write(m_buffer_d, m_count);
    if (rc != ssize_t(m_count)) {
      logging::error("File writing failed: rc %d", rc);
      return;
    }
    m_writing -= 1;
    // reset these to prepare for the new batch
    m_count = 0;
    m_batch_starttime = timestamp;
    printf("*** FileWriter::write: 4 buf %p, cnt %zu\n", m_buffer_d, m_count);
  }

  if (size > (m_bufferSize - m_count)) {
    logging::critical("Buffer size %zu is too small for dgram of size %zu",
                      m_bufferSize - m_count, size);
    abort();
  }
  printf("*** FileWriter::write: 5 buf %p, sz %zu\n", m_buffer_d + m_count, size);
  chkError(cudaMemcpyAsync(m_buffer_d + m_count, devPtr, size, cudaMemcpyDeviceToDevice, m_stream));
  cudaStreamSynchronize(m_stream);
  m_count += size;
  printf("*** FileWriter::write: 6 cnt %zu\n", m_count);
}
