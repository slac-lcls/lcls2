/**
 * @file
 * @brief POSIX shared-memory and message-queue helpers used between the DRP and its Python workers.
 */

#ifndef IPCUTILS_H
#define IPCUTILS_H

#include <sys/ipc.h>

namespace Pds {

/** POSIX shared-memory and message-queue helpers for the DRP. */
namespace Ipc {

/** Open (creating if needed) the POSIX shared memory object key with mode 0666, store its descriptor in shmId and resize it to size bytes. Returns the ftruncate() result, or -1 if shm_open() fails. */
int setupDrpShMem(std::string key, size_t size, int& shmId);
/** Map size bytes of the shared memory descriptor shmId (read-only, or write-only when write is true) and store the address in data. Returns 0, or -1 if mmap() fails; key is not used. */
int attachDrpShMem(std::string key, int& shmId, size_t size, void*& data, bool write);
/** Unmap size bytes at data; returns the munmap() result. */
int detachDrpShMem(void*& data, int size);
/** Open (creating if needed) the POSIX message queue key, write-only or read-only per write, holding one message of up to mqSize bytes, and store its descriptor in mqId. Returns 0, or -1 if mq_open() fails. */
int setupDrpMsgQueue(std::string key, size_t mqSize, int& mqId, bool write);
/** Declared here; IpcUtils.cc defines only detachDrpMq(std::string key), which unlinks the queue, so this signature has no definition. */
int detachDrpMq(void*& data, int size);
/** Send msgsize bytes from msg on queue mqId with priority 31; returns the mq_send() result. */
int drpSend(int mqId, const char *msg, size_t msgsize);
/** Receive one message into msg (up to msgsize bytes) from queue mqId, waiting up to msTmo milliseconds. Returns 0, or -1 on error or timeout. The nanosecond part of the deadline is not normalized, so it can exceed one second. */
int drpRecv(int mqId, char *msg, size_t msgsize, unsigned msTmo);
/** Close shmId if it is non-zero and unlink the shared memory object key; returns the shm_unlink() result. */
int cleanupDrpShmMem(std::string key, int shmId);
/** Close mqId if it is non-zero and unlink the message queue key; returns the mq_unlink() result. */
int cleanupDrpMq(std::string key, int mqId);

}

}

#endif // IPCUTILS_H
