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
      logging::error("CUDA error: %s", str);
      return -2;
    } else {
      logging::error("CUDA error: %d", status.cu_err);
      return -3;
    }
  }
  return 0;
}

static
int chkCuFile(CUfileError_t status)
{
  if (IS_CUFILE_ERR(status.err)) {
    logging::error("cuFile error %d: %s", status.err,
                   CUFILE_ERRSTR(status.err));
    if (status.err == 5011)
      chkCuda(status);
    return -1;
  }
  return chkCuda(status);
}


FileWriter::FileWriter(size_t bufferSize, bool dio) :
  FileWriterBase   (),
  m_fd             (0),
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
}

FileWriter::~FileWriter()
{
  close();

  if (m_stream) {
    if (chkCuFile(cuFileStreamDeregister(m_stream))) {
      logging::error("cuFile unable to deregister CUstream");
    }
  }

  if (m_buffer_d) {
    chkError(cudaFree(m_buffer_d));
  }

  if (chkCuFile(cuFileDriverClose())) {
    logging::critical("Error closing cuFile driver");
    abort();
  }
}

// This must be called from the thread doing the file writing
int FileWriter::registerStream(cudaStream_t stream)
{
  m_stream = stream;

  // Write sizes will generally not be 4K aligned, but
  // all inputs are known at submission time
  if (chkCuFile(cuFileStreamRegister(m_stream,  0x7))) {
    logging::error("cuFile unable to register CUstream");
    return -1;
  }
  return 0;
}

void FileWriter::dumpProperties()
{
  CUfileDrvProps_t devProps;
  if (chkCuFile(cuFileDriverGetProperties(&devProps))) {
    logging::error("Cannot read cuFile capabilities.");
    return;
  }

  logging::info("cuFile properties:");
  logging::info("   Major version:            %u",   devProps.nvfs.major_version);
  logging::info("   Minor version:            %u",   devProps.nvfs.minor_version);
  logging::info("   Poll thresh size:         %zu",  devProps.nvfs.poll_thresh_size);
  logging::info("   Max dir. IO size:         %zu",  devProps.nvfs.max_direct_io_size);
  logging::info("   Driver status flags:    0x%03x", devProps.nvfs.dstatusflags);
  logging::info("     Lustre supported:       %s",   devProps.nvfs.dstatusflags & (1 << CU_FILE_LUSTRE_SUPPORTED)    ? "Yes"     : "No");
  logging::info("     Wekafs supported:       %s",   devProps.nvfs.dstatusflags & (1 << CU_FILE_WEKAFS_SUPPORTED)    ? "Yes"     : "No");
  logging::info("     NFS supported:          %s",   devProps.nvfs.dstatusflags & (1 << CU_FILE_NFS_SUPPORTED)       ? "Yes"     : "No");
  logging::info("     NVMe supported:         %s",   devProps.nvfs.dstatusflags & (1 << CU_FILE_NVME_SUPPORTED)      ? "Yes"     : "No");
  logging::info("   Driver control flags:   0x%02x", devProps.nvfs.dcontrolflags);
  logging::info("     Use poll mode:          %s",   devProps.nvfs.dcontrolflags & (1 << CU_FILE_USE_POLL_MODE)      ? "True"    : "False");
  logging::info("     Allow compat mode:      %s",   devProps.nvfs.dcontrolflags & (1 << CU_FILE_ALLOW_COMPAT_MODE)  ? "Enabled" : "Disabled");

  logging::info("   Feature flags:          0x%02x", devProps.fflags);
  logging::info("     Streams supported:      %s",   devProps.fflags & (1 << CU_FILE_STREAMS_SUPPORTED)     ? "Yes" : "No");
  logging::info("     Batch IO supported:     %s",   devProps.fflags & (1 << CU_FILE_BATCH_IO_SUPPORTED)    ? "Yes" : "No");
  logging::info("     Dyn. routing supported: %s",   devProps.fflags & (1 << CU_FILE_DYN_ROUTING_SUPPORTED) ? "Yes" : "No");
  logging::info("     Parallel IO supported:  %s",   devProps.fflags & (1 << CU_FILE_PARALLEL_IO_SUPPORTED) ? "Yes" : "No");
  logging::info("   Max device cache size:    %u",   devProps.max_device_cache_size);
  logging::info("   Per buffer cache size:    %u",   devProps.per_buffer_cache_size);
  logging::info("   Max pinned memory size:   %u",   devProps.max_device_pinned_mem_size);
  logging::info("   Max batch IO size:        %u",   devProps.max_batch_io_size);
  logging::info("   Max batch IO timeout ms:  %u",   devProps.max_batch_io_timeout_msecs);
}

int FileWriter::open(const std::string& fileName)
{
  int rc;

  if (m_fd > 0) {
    logging::warning("open() closed an already open file");
    close();
  }

  dumpProperties();

  // Open the file
  auto oFlags = O_WRONLY | O_CREAT | O_TRUNC;
  if (m_dio)  oFlags |= O_DIRECT;
  rc = ::open(fileName.c_str(), oFlags, S_IRUSR | S_IRGRP);
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

  // Register the file descriptor with cuFile
  CUfileDescr_t descr;
  memset(reinterpret_cast<void *>(&descr), 0, sizeof(CUfileDescr_t));

  descr.handle.fd = m_fd;
  descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  if ( (rc = chkCuFile(cuFileHandleRegister(&m_handle, &descr))) ) {
    logging::error("Failed to register file handle for fd %d", m_fd);
    close();
    return rc;
  }

  if (m_bufferSize) {
    rc = cudaMalloc(&m_buffer_d, m_bufferSize);
    if (rc != CUDA_SUCCESS) {
      logging::error("Failed to allocate GPU buffer of size %zu: rc %d", m_bufferSize, rc);
      close();
      return rc;
    }

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
  int rc = 0;
  if (m_fd > 0) {
    if (m_count > 0) {
      logging::debug("Flushing %zu bytes to fd %d", m_count, m_fd);
      m_writing += 2;
      _write();
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

  if (m_buffer_d) {
    if (chkCuFile(cuFileBufDeregister(m_buffer_d))) {
      logging::error("Failed to deregister GPU buffer at %p with cuFile", m_buffer_d);
    }
  }
  return rc;
}

ssize_t FileWriter::_write()
{
  printf("*** FileWriter::write: 1, count %zu, fileOffset %zd\n", m_count, m_fileOffset);
  ssize_t rc = cuFileWrite(m_handle, m_buffer_d, m_count, m_fileOffset, 0);
  if (rc < 0) {
    if (IS_CUFILE_ERR(rc))
      logging::error("Write error: buffer %p, count %zu: %s (%zd)", m_buffer_d, m_count, CUFILE_ERRSTR(rc), rc);
    else
      logging::error("Write error: buffer %p, count %zu: %m", m_buffer_d, m_count);
  } else {
    m_fileOffset += rc;
  }
  printf("*** FileWriter::write: 2 buffer %p, count %zu\n", m_buffer_d, m_count);

  return rc;
}

void FileWriter::writeEvent(const void* devPtr, size_t size, const TimeStamp timestamp)
{
  printf("*** FileWriter::writeEvent: 1 ptr %p, sz %zu, t %u.%09u\n", devPtr, size, timestamp.seconds(), timestamp.nanoseconds());

  // triggered only when starting from scratch
  if (m_batch_starttime.value()==0) m_batch_starttime = timestamp;

  // rough calculation: ignore nanoseconds
  unsigned age_seconds = timestamp.seconds()-m_batch_starttime.seconds();
  // write out data if buffer full or batch is too old
  // can't be 1 second without a more precise age calculation, since
  // the seconds field could have "rolled over" since the last event
  printf("*** FileWriter::writeEvent: 2 bufSz %zu, cnt %zu, age %u\n", m_bufferSize, m_count, age_seconds);
  if ((size > (m_bufferSize - m_count)) || age_seconds>2) {
    printf("*** FileWriter::writeEvent: 3 buf %p, cnt %zu\n", m_buffer_d, m_count);
    m_writing += 1;
    auto rc = _write();
    if (rc != ssize_t(m_count)) {
      logging::error("File writing failed: rc %d", rc);
      return;
    }
    m_writing -= 1;
    // reset these to prepare for the new batch
    m_count = 0;
    m_batch_starttime = timestamp;
    printf("*** FileWriter::writeEvent: 4 buf %p, cnt %zu\n", m_buffer_d, m_count);
  }

  if (size > (m_bufferSize - m_count)) {
    logging::critical("Buffer size %zu is too small for dgram of size %zu",
                      m_bufferSize - m_count, size);
    abort();
  }
  printf("*** FileWriter::writeEvent: 5 buf %p, sz %zu\n", m_buffer_d + m_count, size);
  chkError(cudaMemcpyAsync(m_buffer_d + m_count, devPtr, size, cudaMemcpyDeviceToDevice, m_stream));
  m_count += size;
  printf("*** FileWriter::writeEvent: 6 cnt %zu\n", m_count);
}

// ---

FileWriterAsync::FileWriterAsync(size_t bufferSize, bool dio) :
  FileWriter(bufferSize, dio)
{
}

ssize_t FileWriterAsync::_write()
{
  printf("*** FileWriterAsync::write: 1 buffer %p, count %zu vs %zu\n", m_buffer_d, m_count, m_bufferSize);

  off_t bufOffset = 0;         // Always write from the beggining of the buffer
  ssize_t bytesWritten;
  printf("*** FileWriterAsync::write: 2 buf %p, sz %zu, fileOffset %zd, bufOffset %zd\n",
         m_buffer_d, m_count, m_fileOffset, bufOffset);
  if (chkCuFile(cuFileWriteAsync(m_handle, m_buffer_d, &m_count,
                                 &m_fileOffset, &bufOffset,
                                 &bytesWritten, m_stream))) {
    logging::critical("Write error: buffer %p, count %zu", m_buffer_d, m_count);
    abort();
  } else {
    // It seems nonoptimal to synchronize here but attempts to avoid it got messy
    chkError(cudaStreamSynchronize(m_stream));
    if (bytesWritten < 0) {
      if (bytesWritten == -1)
        logging::critical("cuFileWriteAsync IO error: %m");
      else
        logging::critical("cuFileWriteAsync error %zd: %s",
                          bytesWritten, CUFILE_ERRSTR(bytesWritten));
      abort();
    }
    m_fileOffset += bytesWritten;
    printf("*** FileWriterAsync::write: 3 bytesWritten: %zd, fileOffset %zd, count %zd\n",
           bytesWritten, m_fileOffset, m_count);
  }
  printf("*** FileWriterAsync::write: 4 fileOffset %zd\n", m_fileOffset);

  return bytesWritten;
}
