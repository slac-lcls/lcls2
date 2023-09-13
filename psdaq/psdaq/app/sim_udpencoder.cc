//
//  Simulate UDP Encoder
//
#include <stdio.h>
#include <unistd.h>
#include <string>
#include <signal.h>
#include <cmath>

#include "psdaq/tpr/Client.hh"
#include "psdaq/tpr/Frame.hh"

#include "drp/UdpEncoder.hh"

char m_buf[sizeof(Drp::encoder_header_t) + sizeof(Drp::encoder_channel_t)];
bool m_verbose = false;
int m_data_port;
int m_loopbackFd;
int m_dropRequest;
struct sockaddr_in m_loopbackAddr;
uint16_t m_loopbackFrameCount;
static const char *m_addr = "127.0.0.1";
static const double PI = 4.0 * std::atan(1.0);

// forward declarations
void _loopbackInit();
void _loopbackFini();
void _loopbackSend();

static void usage(const char* p) {
  printf("Usage: %s [options]\n",p);
  printf("Options: -r <rate> (0:1Hz,1:10Hz,..)\n");
  printf("         -e <evcode>\n");
  printf("         -p <partition>\n");
  printf("         -a <dest addr>  (default %s)\n", m_addr);
  printf("         -d <data port>  (default %d)\n", Drp::UdpEncoder::DefaultDataPort);
  printf("         -v (verbose)\n");
  printf("Either -r or -e or -p is required\n");
}

static Pds::Tpr::Client* tpr;

Drp::encoder_frame_t  encoder_frame;

static void sigHandler(int signal)
{
  psignal(signal, "\nsim_udpencoder received signal");

  if (signal == SIGUSR1) {
    printf("SIGUSR1: drop 1 frame\n");
    m_dropRequest = 1;
  } else if (signal == SIGUSR2) {
    printf("SIGUSR2: drop 4 frames\n");
    m_dropRequest = 4;
  } else {
    tpr->stop();
    ::exit(signal);
  }
}

int main(int argc, char* argv[])
{
  extern char* optarg;
  char c;

  int rate   = -1;
  int evcode = -1;
  int partition = -1;
  m_data_port = Drp::UdpEncoder::DefaultDataPort;

  while ( (c=getopt( argc, argv, "e:r:p:d:a:vh"))!=EOF) {
    switch(c) {
    case 'e':
      evcode = strtol(optarg,NULL,0);
      break;
    case 'r':
      rate = strtol(optarg,NULL,0);
      break;
    case 'p':
      partition = strtol(optarg,NULL,0);
      break;
    case 'd':
      m_data_port = strtol(optarg,NULL,0);
      break;
    case 'a':
      m_addr = optarg;
      break;
    case 'v':
      m_verbose = true;
      break;
    case 'h':
    default:
      usage(argv[0]);
      return 0;
    }
  }

  if ((rate == -1) && (evcode == -1) && (partition == -1)) {
      // Either -r or -e or -p is required
      usage(argv[0]);
      return 0;
  }

  // create the socket
  _loopbackInit();
  if (m_loopbackFd == -1) {
      printf("_loopbackInit() failed\n");
      return 0;
  }

  // open the timing receiver
  tpr = new Pds::Tpr::Client("/dev/tpra",0);

  struct sigaction sa;
  sa.sa_handler = sigHandler;

  sa.sa_flags = 0;
  sigaction(SIGUSR1,&sa,NULL);
  sigaction(SIGUSR2,&sa,NULL);

  sa.sa_flags = SA_RESETHAND;
  sigaction(SIGINT ,&sa,NULL);
  sigaction(SIGABRT,&sa,NULL);
  sigaction(SIGKILL,&sa,NULL);
  sigaction(SIGSEGV,&sa,NULL);

  if (rate >= 0) {
    tpr->start(Pds::Tpr::TprBase::FixedRate(rate));
  } else if (evcode >= 0) {
    tpr->start(Pds::Tpr::TprBase::EventCode(evcode));
  } else if (partition >= 0) {
    tpr->start(Pds::Tpr::TprBase::Partition(partition));
  }

  if (m_verbose)
    printf("%s: tpr->start() done.\n", __PRETTY_FUNCTION__);

  //  tpr->release();

  while(1) {
    const Pds::Tpr::Frame* frame = tpr->advance();

    if (m_verbose)
      printf("Timestamp %lu.%09u  evc [15:0] 0x%x  evc[47:40] 0x%x\n",
             frame->timeStamp>>32,unsigned(frame->timeStamp&0xffffffff),
             (frame->control[0]>>0)&0xffff,
             (frame->control[2]>>8)&0xff);

    _loopbackSend();    // send frame and increment frame counter

  }

  _loopbackFini();
  return 0;
}

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <fcntl.h>
#include <errno.h>
#include <cstring>

void _loopbackInit()
{
    if (m_verbose) printf("entered %s (port = %d)\n", __PRETTY_FUNCTION__, m_data_port);

    if (m_data_port > 0) {
        m_loopbackFd = socket(AF_INET,SOCK_DGRAM, 0);
        if (m_loopbackFd == -1) {
            perror("socket");
            printf("Error: failed to create loopback socket\n");
        }

        bzero(&m_loopbackAddr, sizeof(m_loopbackAddr));
        m_loopbackAddr.sin_family = AF_INET;
        m_loopbackAddr.sin_addr.s_addr = inet_addr(m_addr);
        m_loopbackAddr.sin_port = htons(m_data_port);
    } else {
        printf("Error: m_data_port = %d in %s\n", m_data_port, __PRETTY_FUNCTION__);
    }

    // prefill buffer with 0
    memset((void *)m_buf, 0, sizeof(m_buf));

    // initialize header
    Drp::encoder_header_t *pHeader = (Drp::encoder_header_t *)m_buf;
    pHeader->frameCount = htons(0);
    pHeader->majorVersion = htons(Drp::UdpEncoder::MajorVersion);
    pHeader->minorVersion = Drp::UdpEncoder::MinorVersion;
    pHeader->microVersion = Drp::UdpEncoder::MicroVersion;
    snprintf(pHeader->hardwareID, 16, "sim_updencoder");
    pHeader->channelMask = pHeader->errorMask = pHeader->mode = 0;

    // initialize channel 0
    Drp::encoder_channel_t *pChannel = (Drp::encoder_channel_t *)(pHeader + 1);
    snprintf(pChannel->hardwareID, 16, "sim_updencoder");
    pChannel->encoderValue = htonl(42);
    pChannel->scale = pChannel->scaleDenom = htons(1);
}

void _loopbackFini()
{
    if (m_verbose) printf("entered %s\n", __PRETTY_FUNCTION__);

    if (m_loopbackFd > 0) {
        if (close(m_loopbackFd)) {
            printf("Error: failed to close loopback socket in %s\n", __PRETTY_FUNCTION__);
        }
    }
}

void _loopbackSend()
{
    int sent = 0;

    if (m_verbose) printf("entered %s\n", __PRETTY_FUNCTION__);

    Drp::encoder_header_t *pHeader = (Drp::encoder_header_t *)m_buf;
    Drp::encoder_channel_t *pChannel = (Drp::encoder_channel_t *)(pHeader + 1);
    uint16_t frameCount = ntohs(pHeader->frameCount);

    if (m_dropRequest > 0) {
        printf("%s: dropping frame #%hu\n", __PRETTY_FUNCTION__, frameCount);
        -- m_dropRequest;
    } else {
        // channel 0 test pattern: 100, 200, 100, 200, 100, ...
        pChannel->encoderValue = htonl((frameCount & 1) ? 200 : 100);
        //// channel 0 test pattern: sine wave between 100 and 200
        //pChannel->encoderValue = htonl(unsigned(100. * (1.5 + 0.5 * std::sin(2. * PI * double(frameCount) / 102.))));
        // send frame
        sent = sendto(m_loopbackFd, (void *)m_buf, sizeof(m_buf), 0,
                      (struct sockaddr *)&m_loopbackAddr, sizeof(m_loopbackAddr));
    }

    // increment frame counter
    pHeader->frameCount = htons(frameCount + 1);

    if (sent == -1) {
        perror("sendto");
        printf("failed to send to loopback socket\n");
    } else {
        if (m_verbose) printf("%s: sent = %d", __PRETTY_FUNCTION__, sent);
    }
}
