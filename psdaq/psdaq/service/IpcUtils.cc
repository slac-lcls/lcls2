#include <assert.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <mqueue.h>
#include <unistd.h>
#include <fcntl.h>  
#include <time.h>
#include <chrono>
#include <cerrno>
#include <string>
#include "psalg/utils/SysLog.hh"
#include "IpcUtils.hh"

using logging = psalg::SysLog;
using ms_t = std::chrono::milliseconds;

namespace Pds::Ipc {

void cleanupIpc(std::string keyBase, int inpMqId, int resMqId, unsigned workerNum)
{
    shm_unlink(("/tmp/shminp_" + keyBase + std::to_string(workerNum)).c_str());
    shm_unlink(("/tmp/shmres_" + keyBase + std::to_string(workerNum)).c_str());
    mq_unlink(("/tmp/mqinp_" + keyBase + std::to_string(workerNum)).c_str());
    mq_close(inpMqId);
    mq_unlink(("/tmp/mqres_" + keyBase + std::to_string(workerNum)).c_str());
    mq_close(resMqId);
}

void cleanupDrpPython(std::string keyBase, int* inpMqId, int* resMqId, unsigned numWorkers)
{
    for (unsigned workerNum=0; workerNum<numWorkers; workerNum++) {
        cleanupIpc(keyBase, inpMqId[workerNum], resMqId[workerNum], workerNum);
    }
}

int setupDrpShMem(std::string key, size_t size, const char* name, int& shmId, unsigned workerNum)
{
    shmId = shm_open(key.c_str(), O_RDWR | O_CREAT, 0666);
    if (shmId == -1) {
        logging::error("[Thread %u] Error in creating Drp %s shared memory for key %s: %m (open step)",
                        workerNum, name, key.c_str());
        return -1;
    }

    int ret = ftruncate(shmId, size);
    if (ret == -1) {
        logging::error("[Thread %u] Error in creating Drp %s shared memory for key %s: %m (ftruncate step)",
                        workerNum, name, key.c_str());
        return -1;
    }

    return 0;
}

int attachDrpShMem(std::string key, const char* name, int& shmId, size_t size, void*& data, bool write, unsigned workerNum)
{

    int prot;
    if (write == true) {
        prot = PROT_WRITE;
    } else {
        prot = PROT_READ;
    }

    data = mmap(NULL, size, prot, MAP_SHARED, shmId, 0);
    if (data == (void *)-1)
    {
        logging::error("[Thread %u] Error attaching Drp %s shared memory for key %u: %m",
                       workerNum, name, key);
        return -1;
    }
    return 0;
}

int setupDrpMsgQueue(std::string key, size_t mqSize, const char* name, int& mqId, bool write, unsigned workerNum)
{

    mq_attr mqattr;
    mqattr.mq_flags = 0;
    mqattr.mq_maxmsg = 1;
    mqattr.mq_msgsize = mqSize;
    mqattr.mq_curmsgs = 0;

    int oflag;
    if (write == true) {
        oflag = O_WRONLY | O_CREAT;
    } else {
        oflag = O_RDONLY | O_CREAT;
    }

    mqId = mq_open(key.c_str(), oflag, 0666, &mqattr);
    if (mqId == -1)
    {
        logging::error("[Thread %u] Error in creating Drp %s message queue with key %u: %m",
                       workerNum, name, key);
        return -1;
    }
    return 0;
}

int drpSend(int mqId, const char *msg, size_t msgsize, unsigned workerNum)
{

    auto rc = mq_send(mqId, msg, msgsize, 31);
    if (rc == -1)
    {
        logging::error("[Thread %u] Error sending message %s to Drp python: %m",
                       workerNum, msg);
        return -1;
    }

    return 0;
}

int drpRecv(int mqId, char *msg, size_t msgsize, unsigned msTmo, unsigned workerNum)
{
    struct timespec t;
    [[maybe_unused]] auto result = clock_gettime(CLOCK_REALTIME, &t);
    assert(result == 0);

    time_t deltasec = (1000000 * (time_t)msTmo) / 1000000000;
    long deltananosec = (1000000 * (time_t)msTmo) % 1000000000;

    t.tv_sec += deltasec;
    t.tv_nsec += deltananosec;

    auto rc = mq_timedreceive(mqId, msg, msgsize, NULL, &t);
    if (rc == -1) {
        if (errno == ETIMEDOUT)
        {
            logging::error("[Thread %u] Message receiving timed out", workerNum);
            return -1;
        } else {
            logging::error("[Thread %u] Error receiving message from Drp python: %m", workerNum);
            return -1;
        }
    }
 
    return 0;
}

}