#ifndef IPCUTILS_H
#define IPCUTILS_H

#include <sys/ipc.h>

namespace Pds {

namespace Ipc {

void cleanupDrpShmMem(std::string key);
void cleanupDrpMq(std::string key, int MqId);
int setupDrpShMem(std::string key, size_t size, int& shmId);
int attachDrpShMem(std::string key, int& shmId, size_t size, void*& data, bool write);
int setupDrpMsgQueue(std::string key, size_t mqSize, int& mqId, bool write);
int drpSend(int mqId, const char *msg, size_t msgsize);
int drpRecv(int mqId, char *msg, size_t msgsize, unsigned msTmo);

}

}

#endif // IPCUTILS_H