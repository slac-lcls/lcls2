#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <zmq.h>

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
    char recvbuf[10];

    // Message Part 1 of 1: command
    if (zmq_send(socket, "Load", 4, 0) == -1) {
        perror("zmq_send");
        return 1;
    }

    printf("Sent 'Load'\n");

    // Reply part 1 of 2: command
    memset(recvbuf, 0, sizeof(recvbuf));
    if ((nbytes = zmq_recv(socket, recvbuf, sizeof(recvbuf), 0)) == -1) {
        perror("zmq_recv");
        return 1;
    } else {
        printf("Received '%s'", recvbuf);
    }

    if (zmq_getsockopt(socket, ZMQ_RCVMORE, &more, &moreSize) != 0) {
        perror("zmq_getsockopt");
        return 1;
    } else if (more) {
        // Reply part 2 of 2: value
        memset(recvbuf, 0, sizeof(recvbuf));
        if ((nbytes = zmq_recv(socket, recvbuf, sizeof(recvbuf), 0)) == -1) {
            perror("zmq_recv");
            return 1;
        } else if (nbytes == 8) {
            int *ii = (int *)recvbuf;
            printf(" [ 0x%08x 0x%08x ]", ii[0], ii[1]);
        } else {
            printf(" [ %d bytes ]", nbytes);
        }
    }
    printf("\n");

    return 0;
}

int store_command(void * socket, int data0, int data1)
{
    int nbytes;
    char recvbuf[10];
    int databuf[2];

    // Message Part 1 of 2: command
    if (zmq_send(socket, "Store", 5, ZMQ_SNDMORE) == -1) {
        perror("zmq_send");
        return 1;
    }

    // Message Part 2 of 2: data
    databuf[0] = data0;
    databuf[1] = data1;
    if (zmq_send(socket, (void *)databuf, sizeof(databuf), 0) == -1) {
        perror("zmq_send");
        return 1;
    }

    printf("Sent 'Store' [ 0x%08x 0x%08x ]\n", data0, data1);

    memset(recvbuf, 0, sizeof(recvbuf));
    if ((nbytes = zmq_recv(socket, recvbuf, sizeof(recvbuf), 0)) == -1) {
        perror("zmq_recv");
        return 1;
    } else {
        printf("Received '%s'\n", recvbuf);
    }

    return 0;
}

int zmq_test(int data0, int data1)
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

    hello_command(reqSocket);
    load_command(reqSocket);
    store_command(reqSocket, data0, data1);
    load_command(reqSocket);

    // clean up
    zmq_close(reqSocket);
    zmq_ctx_destroy (context);

    return 0;
}

int main(int argc, char* argv[])
{
    int rv = -1;

    if (argc != 3) {
        printf("Usage: %s <integer1> <integer2>\n", argv[0]);
    } else {
        rv = zmq_test(strtol(argv[1], NULL, 0), strtol(argv[2], NULL, 0));
    }

    return (rv);
}
