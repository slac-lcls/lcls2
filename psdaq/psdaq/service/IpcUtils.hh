
#ifndef IPCUTILS_H
#define IPCUTILS_H

#include <sys/ipc.h>

namespace Pds {

namespace Ipc {

int setupDrpShMem(std::string key, size_t size, int& shmId);
int attachDrpShMem(std::string key, int& shmId, size_t size, void*& data, bool write);
int detachDrpShMem(void*& data, int size);
int setupDrpMsgQueue(std::string key, size_t mqSize, int& mqId, bool write);
int detachDrpMq(void*& data, int size);
int drpSend(int mqId, const char *msg, size_t msgsize);
int drpRecv(int mqId, char *msg, size_t msgsize, unsigned msTmo);
int cleanupDrpShmMem(std::string key);
int cleanupDrpMq(std::string key, int MqId);

}

}

#endif // IPCUTILS_H