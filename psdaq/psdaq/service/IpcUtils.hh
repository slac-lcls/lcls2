#ifndef IPCUTILS_H
#define IPCUTILS_H

#include <sys/ipc.h>

namespace Pds {

namespace Ipc {

void cleanupDrpPython(std::string keyBase, int* inpMqId, int* resMqId, unsigned numWorkers);
int setupDrpShMem(std::string key, size_t size, const char* name, int& shmId, unsigned workerNum);
int attachDrpShMem(std::string key, const char* name, int& shmId, size_t size, void*& data, bool write, unsigned workerNum);
int setupDrpMsgQueue(std::string key, size_t size, const char* name, int& mqId, bool write, unsigned workerNum);
int drpSend(int mqId, const char *msg, size_t msgsize, unsigned workerNum);
int drpRecv(int mqId, char *msg, size_t msgsize, unsigned msTmo, unsigned workerNum);

}

}

#endif // IPCUTILS_H