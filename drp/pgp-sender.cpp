#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include <stdint.h>
#include <cstdio>
#include <iostream>
#include <chrono>

#include "PgpCardMod.h"

const int SIZE = 700;

int main()
{
    uint32_t port = 1;
    uint32_t vc = 0;

    char dev_name[128];
    snprintf(dev_name, 128, "/dev/pgpcardG3_0_%u", port);

    int fd = open(dev_name, O_RDWR);
    if (fd < 0) {
        std::cout<<"Failed to open pgpcard"<<std::endl;
        return -1;
    }

    PgpCardTx pgp_card;
    pgp_card.model = sizeof(&pgp_card);
    pgp_card.size = SIZE;
    pgp_card.cmd = IOCTL_Normal_Write;
    pgp_card.pgpLane = port - 1;
    pgp_card.pgpVc = vc;
    pgp_card.data = new uint32_t[SIZE];

    int number_of_events = 800000;

    auto start = std::chrono::steady_clock::now();
    for (int n=0; n<number_of_events; n++) {

        float* array = reinterpret_cast<float*>(pgp_card.data);
        int nx = 1;
        int ny = 700;
        double sum = 0.0;
        for (int i=0; i<nx; i++) {
            for (int j=0; j<ny; j++) {
                array[i*ny + j] = n;
                sum += array[i*ny + j];
            }
        }
        //std::cout<<sum / (nx*ny)<<std::endl;
        if (write(fd, &pgp_card, sizeof(PgpCardTx)) < 0) {
            perror("Error writing");
            std::cout<<"Error writing"<<std::endl;
        }
        //std::cout<<"Wrote something"<<std::endl;
        //sleep(1);
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    //std::cout<< duration / double(number_of_events)<<"  ms per message"<<std::endl;
    std::cout<<"PGP sender rate:  "<< double(number_of_events) / duration <<"  kHz"<<std::endl;

    close(fd);
    delete [] pgp_card.data;
}
