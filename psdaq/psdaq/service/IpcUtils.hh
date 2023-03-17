#ifndef IPCUTILS_H
#define IPCUTILS_H

#include <sys/ipc.h>

namespace Pds {

namespace Ipc {


struct Message_t
{
    long mtype;                       // message type
    char mtext[512];                  // message text
};

void cleanupDrpPython(int* inpMqId, int* resMqId, int* inpShmId, int* resShmId, int numWorkers);
int setupDrpShMem(key_t key, size_t size, const char* name, int& shmId, unsigned workerNum);
int attachDrpShMem(key_t key, const char* name, int& shmId, void*& data, unsigned workerNum);
int setupDrpShMem(key_t key, size_t size, const char* name, int& shmId, void*& data, unsigned workerNum);
int setupDrpMsgQueue(key_t key, const char* name, int& mqId, unsigned workerNum);
int send(int mqId, const Message_t& msg, size_t size, unsigned workerNum);
int recv(int mqId, Message_t& msg, unsigned msTmo, clockid_t clockType, unsigned workerNum);

}

}

#endif // IPCUTILS_H