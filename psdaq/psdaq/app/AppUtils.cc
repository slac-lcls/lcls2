#include "AppUtils.hh"

#include <arpa/inet.h>
#include <net/if.h>
#include <netinet/ip.h>
#include <netdb.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <unistd.h>

using namespace Psdaq;

unsigned AppUtils::parse_ip(const char* ipString) {
  unsigned ip = 0;
  in_addr inp;
  if (inet_aton(ipString, &inp)) {
    ip = ntohl(inp.s_addr);
  }
  return ip;
}

unsigned AppUtils::parse_interface(const char* interfaceString) {
  unsigned interface = parse_ip(interfaceString);
  if (interface == 0) {
    int so = socket(AF_INET, SOCK_DGRAM, 0);
    if (so < 0) {
      perror("Failed to open socket\n");
      return 0;
    }
    ifreq ifr;
    strcpy(ifr.ifr_name, interfaceString);
    int rv = ioctl(so, SIOCGIFADDR, (char*)&ifr);
    close(so);
    if (rv != 0) {
      printf("Cannot get IP address for network interface %s.\n",interfaceString);
      return 0;
    }
    interface = ntohl( *(unsigned*)&(ifr.ifr_addr.sa_data[2]) );
  }
  printf("Using interface %s (%d.%d.%d.%d)\n",
         interfaceString,
         (interface>>24)&0xff,
         (interface>>16)&0xff,
         (interface>> 8)&0xff,
         (interface>> 0)&0xff);
  return interface;
}

std::string AppUtils::parse_paddr(unsigned v)
{
  char buff[256];
  if (v==0xffffffff)
    sprintf(buff,"XTPG");
  else if ((v>>24)==0xff) {
    unsigned shelf = (v>>20)&0xf;
    unsigned port  = (v>> 0)&0xff;
    sprintf(buff,"XPM:%d:AMC%d-%d",shelf,port/7,port%7);
  }    
  else
    sprintf(buff,"Unknown");
  return std::string(buff);
}

std::string AppUtils::parse_plink(unsigned v) {
  char chost[32];
  sockaddr_in saddr;
  saddr.sin_addr.s_addr = htonl(0xac150000 | (v&0xffff));
  saddr.sin_port        = 0;
  socklen_t   slen = sizeof(saddr);
  if (getnameinfo(reinterpret_cast<sockaddr*>(&saddr), slen, chost, sizeof(chost), NULL, 0, 0)) {
    printf("lookup of host %08x [%08x] failed\n",saddr.sin_addr.s_addr,v);
    strcpy(chost,"Unknown");
  }
  else {
    sprintf(chost+strlen(chost),":%u",(v>>16)&0xff);
  }
  return std::string(chost);
}

static void* countThread(void* args)
{
  const MonitorArgs* margs = reinterpret_cast<MonitorArgs*>(args);
  std::vector<uint64_t> ovals(margs->values.size());
  std::vector<uint64_t> nvals(margs->values.size());
  std::vector<double  > rates(margs->values.size());
  
  timespec tv;
  clock_gettime(CLOCK_REALTIME,&tv);
  for(unsigned i=0; i<margs->values.size(); i++)
    ovals[i] = *margs->values[i];
  while(1) {
    usleep(1000000);
    timespec otv = tv;
    clock_gettime(CLOCK_REALTIME,&tv);

    double dt     = double( tv.tv_sec - otv.tv_sec) + 1.e-9*(double(tv.tv_nsec)-double(otv.tv_nsec));

    for(unsigned i=0; i<margs->values.size(); i++) {
      nvals[i] = *margs->values[i];
      rates[i] = double(nvals[i]-ovals[i])/dt;
    }

    unsigned rsc=0;
    static const char scchar[] = { ' ', 'k', 'M' };

    for(unsigned i=0; i<margs->values.size(); i++) {
      double rate = rates[i];
      if (rate > 1.e6) {
        rsc     = 2;
        rate   *= 1.e-6;
      }
      else if (rate > 1.e3) {
        rsc     = 1;
        rate   *= 1.e-3;
      }

      printf("%s %7.2f %c%s%s", 
             margs->titles[i].c_str(),
             rate, 
             scchar[rsc], 
             margs->units[i].c_str(),
             (i < margs->values.size()-1) ? ":  " : "\n");
      ovals[i] = nvals[i];
    }
  }
}

pthread_t AppUtils::monitor(const MonitorArgs& args)
{
  pthread_attr_t tattr;
  pthread_attr_init(&tattr);
  pthread_t thr;
  if (pthread_create(&thr, &tattr, &countThread, (void*)&args)) {
    perror("Error creating monitor thread");
  }
  return thr;
}

void MonitorArgs::add(const char* title, const char* unit, uint64_t& value)
{
  titles.push_back(std::string(title));
  units .push_back(std::string(unit));
  values.push_back(&value);
}
