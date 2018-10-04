#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <zmq.h>
#include <fstream>
#include "xtcdata/xtc/Dgram.hh"


class load_xtc{

public:

    char* data;
    size_t fsize;

    load_xtc(){
        FILE *f = fopen("data.xtc", "rb");
        fseek(f, 0, SEEK_END);
        fsize = ftell(f);
        fseek(f, 0, SEEK_SET);

        data = (char *) malloc(fsize);
        fread(data, fsize, 1, f);
        fclose(f);
    }

    ~load_xtc(){
        free(data);
    }
};


int hello_command(void * socket)
{
    int nbytes;
    char recvbuf[10];

    if (zmq_send(socket, "Hello", 5, 0) == -1) {
        perror("zmq_send");
        return 1;
    } else {
        printf("Sent 'Hello'\n");
    }

    memset(recvbuf, 0, sizeof(recvbuf));
    if ((nbytes = zmq_recv(socket, recvbuf, sizeof(recvbuf) - 1, 0)) == -1) {
        perror("zmq_recv");
        return 1;
    } else {
        printf("Received '%s'\n", recvbuf);
    }
    return 0;
}

int load_command(void * socket)
{
    int more;
    size_t moreSize = sizeof(more);
    int nbytes;

    char recvbuf[20];
    size_t dgramLen;
    XtcData::Dgram dgramHeader;

// Message Part 1 of 1: command
    if (zmq_send(socket, "Load", 4, 0) == -1) {
        perror("zmq_send");
        return 1;
    }

    printf("Sent 'Load'\n");

    // Reply part 1 of 3: command
    memset(recvbuf, 0, sizeof(recvbuf));
    if ((nbytes = zmq_recv(socket, recvbuf, sizeof(recvbuf), 0)) == -1) {
        perror("zmq_recv");
        return 1;
    } else {
        printf("Received: %s\n", recvbuf);
    }


    if (zmq_getsockopt(socket, ZMQ_RCVMORE, &more, &moreSize) != 0) {
        perror("zmq_getsockopt");
        return 1;
    } else if (more) {
      nbytes = zmq_recv(socket, &dgramHeader, sizeof(XtcData::Dgram), 0);

      dgramLen = dgramHeader.xtc.extent+sizeof(XtcData::Dgram) - sizeof(XtcData::Xtc);
        // printf("size of dgram is %zu\n", sizeof(XtcData::Dgram));
        // printf("nybtes is %i Dgram length is %zu\n", nbytes, dgramLen);
    }


    char* dgramBuff = (char *) malloc(dgramLen);
    if (zmq_getsockopt(socket, ZMQ_RCVMORE, &more, &moreSize) != 0) {
        perror("zmq_getsockopt");
        return 1;
    } else if (more) {
        // Reply part 3 of 3: value
        if ((nbytes = zmq_recv(socket, dgramBuff, dgramLen, 0)) == -1) {
            perror("zmq_recv");
            return 1;
        } else if (nbytes == dgramLen) {
            printf("Received a dgram containing %zu bytes\n", dgramLen);
            // Do something with it
        } else {
            printf(" %d bytes, dgramlen %d", nbytes, dgramLen);
        }
    }
    printf("\n");
    free(dgramBuff); 
    return 0;
}

int store_command(void * socket, load_xtc& xtc)
{
    int nbytes;


    char recvbuf[10];

    // printf("Size of data buffer %zu\n", xtc.fsize);
    // Message Part 1 of 2: command
    if (zmq_send(socket, "Store", 5, ZMQ_SNDMORE) == -1) {
        perror("zmq_send");
        return 1;
    }


    // Message Part 2 of 2: data
    if (zmq_send(socket, (void *)xtc.data, xtc.fsize, 0) == -1) {
        perror("zmq_send");
        return 1;
    }

    printf("Sent a dgram containing %zu bytes\n", xtc.fsize);

    memset(recvbuf, 0, sizeof(recvbuf));
    if ((nbytes = zmq_recv(socket, recvbuf, sizeof(recvbuf), 0)) == -1) {
        perror("zmq_recv");
        return 1;
    } else {
        printf("Received '%s'\n", recvbuf);
    }

    return 0;
}

int zmq_test(load_xtc& data_buffer)
{
    void* context = zmq_ctx_new();
    if (context == NULL) {
        perror("zmq_ctx_new");
        return 1;
    }
    void* reqSocket = zmq_socket(context, ZMQ_REQ);
    if (reqSocket == NULL) {
        perror("zmq_socket (ZMQ_REQ)");
        return 1;
    }
    if (zmq_connect(reqSocket, "tcp://localhost:5560") == -1) {
        perror("zmq_connect");
        return 1;
    }

    // hello_command(reqSocket);
    load_command(reqSocket);
    store_command(reqSocket, data_buffer);
    load_command(reqSocket);

    // clean up
    zmq_close(reqSocket);
    zmq_ctx_destroy (context);

    return 0;
}


int main(int argc, char* argv[])
{
    int rv = -1;
    load_xtc xtc;

    // if (argc != 3) {
    //     printf("Usage: %s <integer1> <integer2>\n", argv[0]);
    // } else {
    rv = zmq_test(xtc);
    // }
    return (rv);
}
