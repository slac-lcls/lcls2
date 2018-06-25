#include <thread>
#include <cstdio>
#include <cstring>
#include <getopt.h>
#include "PGPReader.hh"
#include "AreaDetector.hh"
#include "Digitizer.hh"
#include "Worker.hh"
#include "Collector.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "xtcdata/xtc/Sequence.hh"
#include <sstream>
#include <iostream>
#include <unistd.h>
#include "zmq.h"

int run_cm_client(std::string addr, std::string name, int platform, std::string cm_host);

using namespace XtcData;

void print_usage(){
    printf("Usage: drp -P <EB server IP address> -i <Contributor ID> -o <Output XTC dir> -d <Device id> -l <Lane mask> -D <Detector type> -u <Name> -p <Platform> -C <Collection mgr host>\n");
    printf("e.g.: sudo psdaq/build/drp/drp -P 172.21.52.128 -i 0 -o /drpffb/yoon82 -d 0x2032 -l 0xf -D Digitizer -u drp -p 0 -C localhost\n");
}

int main(int argc, char* argv[])
{
    Parameters para;
    int device_id = 0x2031;
    int lane_mask = 0xf;
    std::string detector_type;
    const char *name = "drp";
    const char *cm_host = "localhost";
    int platform = 0;
    int c;
    while((c = getopt(argc, argv, "P:i:o:d:l:D:u:p:C:")) != EOF) {
        switch(c) {
            case 'P':
                para.eb_server_ip = optarg;
                break;
            case 'i':
                para.contributor_id = atoi(optarg);
                break;
            case 'o':
                para.output_dir = optarg;
                break;
            case 'd':
                device_id = std::stoul(optarg, nullptr, 16);
                break;
            case 'l':
                lane_mask = std::stoul(optarg, nullptr, 16);
                break;
            case 'D':
                detector_type = optarg;
                break;
            case 'u':
                name = optarg;
                break;
            case 'p':
                platform = atoi(optarg);
                break;
            case 'C':
                cm_host = optarg;
                break;
            default:
                print_usage();
                exit(1);
        }
    }
    printf("eb server ip: %s\n", para.eb_server_ip.c_str());
    printf("contributor id: %u\n", para.contributor_id);
    printf("output dir: %s\n", para.output_dir.c_str());

    printf("Calling run_cm_client(\"%s\", \"%s\", %d, \"%s\")...\n",
           para.eb_server_ip.c_str(), name, platform, cm_host);
    run_cm_client(para.eb_server_ip, name, platform, cm_host);

    Factory<Detector> f;
    f.register_type<Digitizer>("Digitizer");
    f.register_type<AreaDetector>("AreaDetector");
    Detector* d = f.create(detector_type.c_str());

    int num_workers = 2;
    int num_entries = 131072;
    MemPool pool(num_workers, num_entries);
    PGPReader pgp_reader(pool, device_id, lane_mask, num_workers);
    std::thread pgp_thread(&PGPReader::run, std::ref(pgp_reader));
    pin_thread(pgp_thread.native_handle(), 1);

    // event builder
    Pds::StringList peers;
    peers.push_back(para.eb_server_ip);
    Pds::StringList ports;
    ports.push_back("32768");
    Pds::Eb::EbLfClient myEbLfClient(peers, ports);
    MyBatchManager myBatchMan(myEbLfClient, para.contributor_id);
    unsigned timeout = 10;
    int ret = myEbLfClient.connect(para.contributor_id, timeout,
                                   myBatchMan.batchRegion(), 
                                   myBatchMan.batchRegionSize());
    if (ret) {
        printf("ERROR in connecting to event builder!!!!\n");
    }

    // start performance monitor thread
    std::thread monitor_thread(monitor_func, std::ref(pgp_reader.get_counters()),
                               std::ref(pool), std::ref(myBatchMan));

    // start worker threads
    std::vector<std::thread> worker_threads;
    for (int i = 0; i < num_workers; i++) {
        worker_threads.emplace_back(worker, d, std::ref(pool.worker_input_queues[i]),
                                    std::ref(pool.worker_output_queues[i]), i);
        pin_thread(worker_threads[i].native_handle(), 2 + i);
    }

    collector(pool, para, myBatchMan);

    pgp_thread.join();
    for (int i = 0; i < num_workers; i++) {
        worker_threads[i].join();
    }

    // shutdown monitor thread
    // counter->total_bytes_received = -1;
    //p.exchange(counter, std::memory_order_release);
    // monitor_thread.join();
}

#include "psdaq/service/Task.hh"
#include "psdaq/service/TaskObject.hh"
#include "psdaq/service/Routine.hh"
#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"

using namespace rapidjson;

// FIXME set drp level
#define HELLO_FORMAT "{\"level\":4,\"host\":\"%s\",\"pid\":%d,\"name\":\"%s\"}"

    class CMClient : public Pds::Routine
    {
    public:
      CMClient(std::string addr, std::string name, int platform, std::string cm_host, void *context) :
        _addr(addr),
        _name(name),
        _platform(platform),
        _cm_host(cm_host),
        _context(context)
        {}
      ~CMClient()        {}
      // Routine interface
      void routine(void);

      enum { PORT_BASE = 29980 };

      int dealer_port();
      int sub_port();

    private:
      std::string _addr;
      std::string _name;
      int _platform;
      std::string _cm_host;
      void * _context;
    };

#define KEYMAX  40

    int CMClient::dealer_port()
    {
        return (PORT_BASE + _platform);
    }

    int CMClient::sub_port()
    {
        return (PORT_BASE + _platform + 10);
    }

    void CMClient::routine()
    {
        int len, more, ii, gotConnect;
        int zero = 0;
        size_t olen;
        char buf[1024];
        char keybuf[KEYMAX];
        zmq_pollitem_t items[2];

        printf(" ** %s: addr=\"%s\"\n", __PRETTY_FUNCTION__, _addr.c_str());
        printf(" ** name=\"%s\"\n", _name.c_str());
        printf(" ** platform=%d\n", _platform);
        printf(" ** cm_host=\"%s\"\n", _cm_host.c_str());
        printf(" ** context=%p\n", _context);

        // create and connect PAIR socket 
        void *xmitter = zmq_socket (_context, ZMQ_PAIR);
        zmq_connect (xmitter, "inproc://pair");

        // create and connect DEALER socket 
        void *dealer = zmq_socket(_context, ZMQ_DEALER);
        if (zmq_setsockopt(dealer, ZMQ_LINGER, &zero, sizeof(zero)) == -1) {
            perror("zmq_setsockopt(ZMQ_LINGER)");
        }

        // collection mgr port determined by platform
        sprintf(buf, "tcp://%s:%d", _cm_host.c_str(), dealer_port());
        if (zmq_connect(dealer, buf) == -1) {
            perror("zmq_connect");
        }

        // create and connect SUB socket 
        void* sub = zmq_socket(_context, ZMQ_SUB);

        // subscribe to all mesages
        if (zmq_setsockopt(sub, ZMQ_SUBSCRIBE, "", 0) == -1) {
            perror("zmq_setsockopt(ZMQ_SUBSCRIBE)");
        }
        if (zmq_setsockopt(sub, ZMQ_LINGER, &zero, sizeof(zero)) == -1) {
            perror("zmq_setsockopt(ZMQ_LINGER)");
        }

        // port determined by platform
        sprintf(buf, "tcp://%s:%d", _cm_host.c_str(), sub_port());
        if (zmq_connect(sub, buf) == -1) {
            perror("zmq_connect");
        }

        // prepare HELLO message
        char hello_msg[512];
        char hostbuf[64];
        if (gethostname(hostbuf, sizeof(hostbuf)) == -1) {
            perror("gethostname");
            sprintf(hostbuf, "%s", "unknown");
        }
        int pid = getpid();
        snprintf(hello_msg, sizeof(hello_msg), HELLO_FORMAT, hostbuf, pid, _name.c_str());

        // set up for polling two ZMQ sockets
        items[0].socket = dealer;
        items[0].events = ZMQ_POLLIN;
        items[1].socket = sub;
        items[1].events = ZMQ_POLLIN;

        while (1) {
            // Poll for events indefinitely
            if (zmq_poll (items, 2, -1) == -1) {
                perror("zmq_poll");
                break;
            }
            // check for event on DEALER socket
            if (items[0].revents) {
                gotConnect = 0;
                for (more = 1, ii = 0; (more == 1) && (ii < 10); ii++) {

                    // receive one part
                    len = zmq_recv(dealer, buf, sizeof(buf), 0);
                    if (len == -1) {
                        perror("zmq_recv");
                        goto cm_task_shutdown;
                    }
                    if (ii == 0) {
                        if (len >= KEYMAX) {
                            // allow room for NULL
                            len = KEYMAX - 1;
                        }
                        strncpy(keybuf, buf, len);
                        keybuf[len] = '\0';
                        printf("Received ZMQ Message on DEALER socket: %s\n", keybuf);
                        if (strcmp(keybuf, "ALLOC") == 0) {
                            printf("Got ALLOC\n");
                        } else if (strcmp(keybuf, "CONNECT") == 0) {
                            printf("Got CONNECT\n");
                            gotConnect = 1;
                        }
                    }
                    if ((ii == 4) && gotConnect) {
                        // forward CONNECT message to PAIR socket
                        zmq_send (xmitter, buf, len, 0);
                        goto cm_task_shutdown;
                    }

                    // check for more parts
                    olen = sizeof(more);
                    if (zmq_getsockopt(dealer, ZMQ_RCVMORE, &more, &olen) == -1) {
                        perror("zmq_getsockopt");
                        goto cm_task_shutdown;
                    }
                }
            }
            // check for event on SUB socket
            if (items[1].revents) {
                for (more = 1, ii = 0; (more == 1) && (ii < 10); ii++) {

                    // receive one part
                    len = zmq_recv(sub, buf, sizeof(buf), 0);
                    if (len == -1) {
                        perror("zmq_recv");
                        goto cm_task_shutdown;
                    }
                    if (ii == 0) {
                        if (len >= KEYMAX) {
                            // allow room for NULL
                            len = KEYMAX - 1;
                        }
                        strncpy(keybuf, buf, len);
                        keybuf[len] = '\0';
                        printf("Received ZMQ Message on SUB socket: %s\n", keybuf);
                        if (strcmp(keybuf, "PING") == 0) {
                            if (zmq_send (dealer, "PONG", 4, 0) == -1) {
                                perror("zmq_send");
                                goto cm_task_shutdown;
                            } else {
                                printf("Sent PONG to DEALER socket\n");
                            }
                        } else if (strcmp(keybuf, "PLAT") == 0) {
                            if (zmq_send (dealer, "HELLO", 5, ZMQ_SNDMORE) == -1) {
                                perror("zmq_send");
                                goto cm_task_shutdown;
                            }
                            zmq_send(dealer, "", 0, ZMQ_SNDMORE);              // msg[2]
                            zmq_send(dealer, "", 0, ZMQ_SNDMORE);              // msg[3]
                            zmq_send(dealer, hello_msg, strlen(hello_msg), 0); // msg[4]
                            printf("Sent HELLO to DEALER socket\n");
                        } else if (strcmp(keybuf, "DIE") == 0) {
                            // forward DIE to PAIR socket
                            zmq_send (xmitter, keybuf, 3, 0);
                            goto cm_task_shutdown;
                        }
                    }

                    // check for more parts
                    olen = sizeof(more);
                    if (zmq_getsockopt(sub, ZMQ_RCVMORE, &more, &olen) == -1) {
                        perror("zmq_getsockopt");
                        goto cm_task_shutdown;
                    }
                }
            }
        } // end of while

cm_task_shutdown:

        // clean up
        if (zmq_close(sub)) {
            perror("zmq_close(sub)");
        }
        if (zmq_close(dealer)) {
            perror("zmq_close(dealer)");
        }
        if (zmq_close(xmitter)) {
            perror("zmq_close(xmitter)");
        }
        printf(" ** collection mgr client task shutdown **\n");
    }

int run_cm_client(std::string addr, std::string name, int platform, std::string cm_host)
{
    char buf[512];

    // create zmq context
    void* context = zmq_ctx_new();
    if (!context) {
        perror("zmq_ctx_new");
        return 1;
    }

    // bind inproc socket before starting client task
    void *receiver = zmq_socket (context, ZMQ_PAIR);
    zmq_bind (receiver, "inproc://pair");

    // create collection mgr client routine
    CMClient *_cmClient = new CMClient(addr, name, platform, cm_host, context);

    // create collection mgr client task
    Pds::Task *_cmTask = new Pds::Task(Pds::TaskObject("cmclient"));

    // create collection mgr client thread
    _cmTask->call(_cmClient);

    printf(" ** wait for msg from collection mgr client task **\n");
    int len = zmq_recv(receiver, buf, sizeof(buf), 0);
    if (len == -1) {
        perror("zmq_recv");
    } else {
        printf("Received msg of len %d on PAIR socket\n", len);
        buf[len] = '\0';    // add NULL termination
        printf("\nRaw msg:\n");
        printf("--------------------------------------------------------------\n");
        printf("%s\n", buf);
        printf("--------------------------------------------------------------\n");
        // parse JSON into a Document
        rapidjson::Document document;
        document.Parse(buf);

        if (!document.IsObject()) {
            printf("JSON: object not found\n");
        } else {
            // use JSON PrettyWriter
            printf("\nJSON msg:\n");
            printf("--------------------------------------------------------------\n");
            StringBuffer sb;
            PrettyWriter<StringBuffer> writer(sb);
            document.Accept(writer);    // Accept() traverses the DOM and generates Handler events.
            puts(sb.GetString());
            printf("--------------------------------------------------------------\n");

            // interpret Document
            if (document.HasMember("msgType") && document.HasMember("msgVer")) {
                printf("JSON: msgType=\"%s\" msgVer=%d\n",
                       document["msgType"].GetString(),
                       document["msgVer"].GetInt());

                // if the ALLOC step was not completed, ports entry will be missing
                if (document["procs"]["control"].HasMember("ports")) {
                    // get references to pull_port adrs and port
                    rapidjson::Value& adrs = document["procs"]["control"]["ports"]["pull_port"]["adrs"];
                    rapidjson::Value& port = document["procs"]["control"]["ports"]["pull_port"]["port"];
                    printf("JSON: pull_port adrs=\"%s\" port=%d\n", adrs.GetString(), port.GetInt());
                } else {
                    printf("JSON: control ports not found\n");
                }
            } else {
                printf("JSON: msgType and msgVer not found\n");
            }
        }
    }

    // clean up
    // FIXME possible race condition with thread closing its xmitter socket
    if (zmq_close(receiver)) {
        perror("zmq_close");
    }
    if (zmq_ctx_term(context)) {
        perror("zmq_ctx_term");
    }

    return 0;
}
