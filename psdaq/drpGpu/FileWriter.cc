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


// For dealing with the errors in cuFile structs
static
int checkError(CUfileError_t status, const char* const func, const char* const file,
               const int line, const bool crash, const char* const msg)
{
  if (IS_CUFILE_ERR(status.err)) {
    if (IS_CUDA_ERR(status)) {
      checkError(status.cu_err, func, file, line, crash, msg);
    } else {
      logging::error("%s:%d:  %s (%i): '%s' %s",
                     file, line, "cuFile error", status.err, CUFILE_ERRSTR(status.err), msg);
      if (crash)  abort();
    }
    return -1;
  }
  return checkError(status.cu_err, func, file, line, crash, msg);
}


FileWriter::FileWriter(size_t bufferSize, bool dio) :
  FileWriterBase   (),
  m_stream         (0),
  m_batch_starttime(0, 0),
  m_buffer_d       (nullptr),
  m_fileOffset     (0),
  m_bufferSize     (bufferSize),
  m_fd             (0),
  m_dio            (dio),
  m_count          (0)
{
  if (bufferSize & (bufferSize - 1)) {
    logging::critical("cuFile buffer size must be a power of 2; got %zu\n", bufferSize);
    exit(EXIT_FAILURE);
  }

  if (chkError(cuFileDriverOpen())) {
    logging::critical("Error opening cuFile driver");
    exit(EXIT_FAILURE);
  }

  CUfileDrvProps_t devProps;
  if (chkError(cuFileDriverGetProperties(&devProps))) {
    logging::error("Cannot read cuFile capabilities.");
    return;
  }
  if (m_bufferSize > devProps.max_device_pinned_mem_size) {
    logging::warning("cuFile Buffer size is limited to max_device_pinned_mem_size %u; request was %zu",
                     devProps.max_device_pinned_mem_size, m_bufferSize);
    m_bufferSize = devProps.max_device_pinned_mem_size;
  }

  if (chkError(cudaMalloc(&m_buffer_d, m_bufferSize))) {
    logging::error("Failed to allocate GPU buffer of size %zu", m_bufferSize);
    m_buffer_d = nullptr;
  }
  logging::debug("FileWriter: cuFile buffer: %p, size %zu\n", m_buffer_d, m_bufferSize);
}

FileWriter::~FileWriter()
{
  printf("*** FileWriter::dtor 1\n");

  close();

  if (m_stream) {
    printf("*** FileWriter::dtor 2\n");
    if (chkError(cuFileStreamDeregister(m_stream))) {
      logging::error("cuFile unable to deregister CUstream");
    }
  }

  if (m_buffer_d) {
    printf("*** FileWriter::dtor 3\n");
    chkError(cudaFree(m_buffer_d));
    m_buffer_d = nullptr;
  }

  printf("*** FileWriter::dtor 4\n");
  if (chkError(cuFileDriverClose())) {
    logging::critical("Error closing cuFile driver");
    exit(EXIT_FAILURE);
  }
  printf("*** FileWriter::dtor 5\n");
}

// This must be called from the thread doing the file writing
int FileWriter::registerStream(cudaStream_t stream)
{
  m_stream = stream;

  // Write sizes will generally not be 4K aligned, but
  // all inputs are known at submission time
  if (chkError(cuFileStreamRegister(m_stream,  0x7))) {
    logging::error("cuFile unable to register CUstream");
    return -1;
  }
  return 0;
}

void FileWriter::dumpProperties()
{
  CUfileDrvProps_t devProps;
  if (chkError(cuFileDriverGetProperties(&devProps))) {
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
  auto oFlags = O_CREAT | O_WRONLY | O_TRUNC;
  if (m_dio)  oFlags |= O_DIRECT;
  rc = ::open(fileName.c_str(), oFlags, S_IRUSR | S_IWUSR | S_IRGRP); // W is required
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
  memset(static_cast<void *>(&descr), 0, sizeof(descr));

  descr.handle.fd = m_fd;
  descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  if ( (rc = chkError(cuFileHandleRegister(&m_handle, &descr))) ) {
    logging::error("Failed to register file handle for fd %d", m_fd);
    close();
    return rc;
  }

  if (m_bufferSize) {
    if ( (rc = chkError(cuFileBufRegister(m_buffer_d, m_bufferSize, 0))) ) {
      logging::error("Failed to register GPU buffer %p, size %zu with cuFile",
                     m_buffer_d, m_bufferSize);
      close();
      return rc;
    }
  }

  _reset();

  return rc;
}

int FileWriter::close()
{
  int rc = 0;
  if (m_fd > 0) {
    _flush();
    cuFileHandleDeregister(m_handle);
    logging::debug("Closing fd %d", m_fd);
    rc = ::close(m_fd);
    if (rc == -1) {
      // %m will be replaced by the string strerror(errno)
      logging::error("Error closing fd %d: %m", m_fd);
    }
    m_fd = 0;

    if (chkError(cuFileBufDeregister(m_buffer_d))) {
      logging::error("Failed to deregister GPU buffer at %p with cuFile", m_buffer_d);
    }
  }

  return rc;
}

void FileWriter::_reset()
{
  m_count           = 0;
  m_fileOffset      = 0;
  m_batch_starttime = XtcData::TimeStamp(0,0);
}

void FileWriter::_flush()
{
  logging::debug("FileWriter flushing %zu bytes to fd %d", m_count, m_fd);
  m_writing += 2;
  _write();
  m_writing -= 2;
  m_count = 0;
  m_batch_starttime = XtcData::TimeStamp(0,0);
}

ssize_t FileWriter::_write()
{
  ssize_t rc = 0;
  if (m_count) {
    rc = cuFileWrite(m_handle, m_buffer_d, m_count, m_fileOffset, 0);
    if (rc < 0) {
      if (IS_CUFILE_ERR(rc))
        logging::error("Write error: buffer %p, count %zu: %s (%zd)", m_buffer_d, m_count, CUFILE_ERRSTR(rc), rc);
      else
        logging::error("Write error: buffer %p, count %zu: %m", m_buffer_d, m_count);
    } else {
      m_fileOffset += rc;
    }
  }

  return rc;
}

void FileWriter::writeEvent(const void* devPtr, size_t size, const TimeStamp timestamp)
{
  // triggered only when starting from scratch
  if (m_batch_starttime.value()==0) m_batch_starttime = timestamp;

  // rough calculation: ignore nanoseconds
  unsigned age_seconds = timestamp.seconds()-m_batch_starttime.seconds();
  // write out data if buffer full or batch is too old
  // can't be 1 second without a more precise age calculation, since
  // the seconds field could have "rolled over" since the last event
  if ((size > (m_bufferSize - m_count)) || age_seconds>2) {
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
  }

  if (size > (m_bufferSize - m_count)) {
    logging::critical("Buffer size %zu is too small for dgram of size %zu",
                      m_bufferSize - m_count, size);
    exit(EXIT_FAILURE);
  }
  //printf("*** writeEvent: buffer %p + count %zu: %p, devPtr %p, sz %zu\n",
  //       m_buffer_d, m_count, m_buffer_d + m_count, devPtr, size);
  chkFatal(cudaMemcpyAsync(m_buffer_d + m_count, devPtr, size, cudaMemcpyDeviceToDevice, m_stream));
  m_count += size;
}

// ---

FileWriterAsync::FileWriterAsync(size_t bufferSize, bool dio) :
  FileWriter    (2 * bufferSize, dio),
  m_counts      {0, 0},
  m_bufOffset   {0, off_t(bufferSize)},
  m_index       (0),
  m_bytesWritten(0)
{
}

void FileWriterAsync::_reset()
{
  m_counts[0]       = 0;
  m_counts[1]       = 0;
  m_index           = 0;
  m_bytesWritten    = 0;
  m_fileOffset      = 0;
  m_batch_starttime = XtcData::TimeStamp(0,0);
}

void FileWriterAsync::_flush()
{
  logging::debug("FileWriterAsync flushing %zu bytes to fd %d", m_counts[m_index], m_fd);
  m_writing += 2;
  _write();
  m_writing -= 2;

  // Wait for the current write to complete and check its completion status
  chkError(cudaStreamSynchronize(m_stream));
  if (m_bytesWritten < 0) {
    if (m_bytesWritten == -1)
      logging::critical("cuFileWriteAsync IO error: %m");
    else
      logging::critical("cuFileWriteAsync error %zd: %s",
                        m_bytesWritten, CUFILE_ERRSTR(m_bytesWritten));
    exit(EXIT_FAILURE);
  }
  m_fileOffset += m_bytesWritten;

  m_counts[m_index] = 0;
  m_batch_starttime = XtcData::TimeStamp(0,0);
}

void FileWriterAsync::_write()
{
  // Wait for the previous write to complete and check its completion status
  chkError(cudaStreamSynchronize(m_stream));
  if (m_bytesWritten < 0) {
    if (m_bytesWritten == -1)
      logging::critical("cuFileWriteAsync IO error: %m");
    else
      logging::critical("cuFileWriteAsync error %zd: %s",
                        m_bytesWritten, CUFILE_ERRSTR(m_bytesWritten));
    exit(EXIT_FAILURE);
  }
  m_fileOffset += m_bytesWritten;

  if (m_counts[m_index]) {
    if (chkError(cuFileWriteAsync(m_handle, m_buffer_d, &m_counts[m_index],
                                   &m_fileOffset, &m_bufOffset[m_index],
                                   &m_bytesWritten, m_stream))) {
      logging::critical("Write error: buffer %p, count %zu", m_buffer_d, m_counts[m_index]);
      exit(EXIT_FAILURE);
    }
  }

  m_index = (m_index + 1) & 0x1;
}

void FileWriterAsync::writeEvent(const void* devPtr, size_t size, const TimeStamp timestamp)
{
  // triggered only when starting from scratch
  if (m_batch_starttime.value()==0) m_batch_starttime = timestamp;

  // rough calculation: ignore nanoseconds
  unsigned age_seconds = timestamp.seconds()-m_batch_starttime.seconds();
  // write out data if buffer full or batch is too old
  // can't be 1 second without a more precise age calculation, since
  // the seconds field could have "rolled over" since the last event
  auto bufferSize = m_bufferSize / 2;   // For 2 ping pong buffers
  if ((size > (bufferSize - m_counts[m_index])) || age_seconds>2) {
    // Start a new write
    m_writing += 1;
    _write();
    m_writing -= 1;

    // reset these to prepare for the new batch
    m_counts[m_index] = 0;
    m_batch_starttime = timestamp;
  }

  if (size > (bufferSize - m_counts[m_index])) {
    logging::critical("Buffer size %zu is too small for dgram of size %zu",
                      bufferSize - m_counts[m_index], size);
    exit(EXIT_FAILURE);
  }
  //printf("*** writeEvent: buffer %p + offset[%lu] %zu + counts[%lu] %zu: %p, devPtr %p, sz %zu\n",
  //       m_buffer_d, m_index, m_bufOffset[m_index], m_index, m_counts[m_index],
  //       m_buffer_d + m_bufOffset[m_index] + m_counts[m_index], devPtr, size);
  chkError(cudaMemcpyAsync(m_buffer_d + m_bufOffset[m_index] + m_counts[m_index], devPtr, size, cudaMemcpyDeviceToDevice, m_stream));
  m_counts[m_index] += size;
}
