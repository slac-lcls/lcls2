#include <assert.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/msg.h>
#include <time.h>
#include <chrono>
#include <cerrno>
#include "psalg/utils/SysLog.hh"
#include "IpcUtils.hh"

using logging = psalg::SysLog;
using ms_t = std::chrono::milliseconds;

namespace Pds::Ipc {

void cleanupIpc(int inpMqId, int resMqId, int inpShmId, int resShmId, int workerNum)
{
    if (inpMqId) {
        msgctl(inpMqId, IPC_RMID, NULL);
        inpMqId  = 0;
    }
    if (resMqId)   {
        msgctl(resMqId, IPC_RMID, NULL); 
        resMqId  = 0;
    }
    if (inpShmId) {
        shmctl(inpShmId, IPC_RMID, NULL);
        inpShmId = 0;
    }
    if (resShmId) {
        shmctl(resShmId, IPC_RMID, NULL); 
        resShmId = 0;
    }
}

void cleanupDrpPython(int* inpMqId, int* resMqId, int* inpShmId, int* resShmId, int numWorkers)
{
    for (int workerNum=0; workerNum<numWorkers; workerNum++) {
        cleanupIpc(inpMqId[workerNum], resMqId[workerNum], inpShmId[workerNum], resShmId[workerNum], workerNum);
    }
}

int setupDrpShMem(key_t key, size_t size, const char* name, int& shmId, unsigned workerNum)
{

    shmId = shmget(key, size, IPC_CREAT | 0666); // IPC_EXCL
    if (shmId == -1)
    {
        logging::error("[Thread %u] Error in creating Drp %s shared memory for key %u: %m",
                        workerNum, name, key);
        return -1;
    }
    return 0;
}

int attachDrpShMem(key_t key, const char* name, int& shmId, void*& data, unsigned workerNum)
{
    data = shmat(shmId, nullptr, 0);
    if (data == (void *)-1)
    {
        logging::error("[Thread %u] Error attaching Drp %s shared memory for key %u: %m",
                       workerNum, name, key);
        return -1;
    }
    return 0;
}

int setupDrpMsgQueue(key_t key, const char* name, int& mqId, unsigned workerNum)
{
    mqId = msgget(key, IPC_CREAT | 0666);
    if (mqId == -1)
    {
        logging::error("[Thread %u] Error in creating Drp %s message queue with key %u: %m",
                       workerNum, name, key);
        return -1;
    }
    return 0;
}

int send(int mqId, const Message_t& msg, size_t size, unsigned workerNum)
{
    int rc = msgsnd(mqId, (void *)&msg, size, 0);
    if (rc == -1)
    {
        logging::error("[Thread %u] Error sending message '%c' to Drp python: %m",
                       workerNum, msg.mtext[0]);
        return -1;
    }
    return 0;
}

int recv(int mqId, Message_t& msg, unsigned msTmo, clockid_t clockType, unsigned workerNum)
{

    struct timespec t;
    [[maybe_unused]] auto result = clock_gettime(clockType, &t);
    assert(result == 0);
    auto tp = std::chrono::seconds(t.tv_sec) + std::chrono::nanoseconds(t.tv_nsec);

    while (true)
    {
        auto rc = msgrcv(mqId, &msg, sizeof(msg.mtext), 0, IPC_NOWAIT);
        if (rc != -1)  break;

        if (errno != ENOMSG)
        {
            logging::error("[Thread %u] Error receiving message from Drp python: %m", workerNum);
            return -1;
        }

        result = clock_gettime(clockType, &t);
        assert(result == 0);
        auto now = std::chrono::seconds(t.tv_sec) + std::chrono::nanoseconds(t.tv_nsec);
        
        auto dt  = std::chrono::duration_cast<ms_t>(now - tp).count();
        
        if (dt > msTmo)
        {
            logging::error("[Thread %u] Message receiving timed out", workerNum);
            return -1;
        }
    }
    return 0;
}

}