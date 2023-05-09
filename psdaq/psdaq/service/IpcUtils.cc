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

void cleanupDrpShmMem(std::string key)
{
    shm_unlink(key.c_str());
}

void cleanupDrpMq(std::string key, int MqId)
{
    mq_unlink(key.c_str());
    mq_close(MqId);
}

int setupDrpShMem(std::string key, size_t size, int& shmId)
{
    shmId = shm_open(key.c_str(), O_RDWR | O_CREAT, 0666);
    if (shmId == -1) {
        return -1;
    }

    int ret = ftruncate(shmId, size);
    return ret;
}

int attachDrpShMem(std::string key, int& shmId, size_t size, void*& data, bool write)
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
        return -1;
    }
    return 0;
}

int setupDrpMsgQueue(std::string key, size_t mqSize, int& mqId, bool write)
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
        return -1;
    }
    return 0;
}

int drpSend(int mqId, const char *msg, size_t msgsize)
{
    auto rc = mq_send(mqId, msg, msgsize, 31);
    return rc;
}

int drpRecv(int mqId, char *msg, size_t msgsize, unsigned msTmo)
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
            logging::critical("Message receiving timed out");
            return -1;
        } else {
            return -1;
        }
    }
    return 0;
}

}