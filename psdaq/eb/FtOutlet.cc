#include "FtOutlet.hh"

#include "Endpoint.hh"

#include <rdma/fi_rma.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>                     // For sleep()...

using namespace Pds;
using namespace Pds::Fabrics;

static const size_t scratch_size = sizeof(Fabrics::RemoteAddress);


FtOutlet::FtOutlet(StringList&  remote,
                   std::string& port) :
  _remote (remote),
  _port   (port),
  _ep     (remote.size()),
  _mr     (remote.size()),
  _ra     (remote.size()),
  _scratch(new char[scratch_size])
{
}

FtOutlet::~FtOutlet()
{
  unsigned nMr = _mr.size();
  unsigned nEp = _ep.size();

  for (unsigned i = 0; i < nMr; ++i)
    if (_mr[i])   delete _mr[i];
  for (unsigned i = 0; i < nEp; ++i)
    if (_ep[i])   delete _ep[i];
}

int FtOutlet::connect(size_t   size,
                      unsigned tmo)
{
  for (unsigned i = 0; i < _remote.size(); ++i)
  {
    int ret = _connect(_remote[i], _port, size, _ep[i], _mr[i], _ra[i], tmo);
    if (ret)
    {
      fprintf(stderr, "_connect() failed at index %u\n", i);
      return ret;
    }

    printf("Peer %d connected: pool @ %p, size: %zd\n", i, (void*)_ra[i].addr, _ra[i].extent);
  }

  return 0;
}

int FtOutlet::_connect(std::string&   remote,
                       std::string&   port,
                       size_t         size,
                       Endpoint*&     ep,
                       MemoryRegion*& mr,
                       RemoteAddress& ra,
                       unsigned       tmo)
{
  ep = new Endpoint(remote.c_str(), port.c_str());
  if (!ep || (ep->state() != EP_UP))
  {
    fprintf(stderr, "Failed to initialize fabrics endpoint %s:%s: %s\n",
            remote.c_str(), port.c_str(), ep->error());
    perror("new Endpoint");
    return ep ? ep->error_num() : -1;
  }

  Fabric* fab = ep->fabric();

  mr = fab->register_memory(_scratch, scratch_size);
  if (!mr)
  {
    fprintf(stderr, "Failed to register memory region @ %p, sz %zu: %s\n",
            _scratch, scratch_size, fab->error());
    perror("fab->register_memory");
    return fab->error_num();
  }

  bool tmoEnabled = tmo != 0;
  while (!ep->connect() && (!tmoEnabled || --tmo))
  {
    sleep (1);
  }
  if (tmoEnabled && (tmo == 0))
  {
    fprintf(stderr, "Failed to connect endpoint %s:%s: %s\n",
            remote.c_str(), port.c_str(), ep->error());
    perror("ep->connect()");
    return -1;
  }

  if (!ep->recv_sync(_scratch, sizeof(ra), mr))
  {
    fprintf(stderr, "Failed receiving remote memory specs from server: %s\n",
            ep->error());
    perror("recv RemoteAddress");
    return ep->error_num();
  }

  memcpy(&ra, _scratch, sizeof(ra));
  if (size > ra.extent)
  {
    fprintf(stderr, "Remote pool size (%lu) is less than local pool size (%lu)\n",
            ra.extent, size);
    return -1;
  }

  return 0;
}

int FtOutlet::shutdown()
{
  int ret = FI_SUCCESS;

  //for (unsigned i = 0; i < _ep.size(); ++i)
  //{
  //  // Revisit: What's needed here?
  //}

  return ret;
}

int FtOutlet::post(fi_msg_rma* msg,
                   unsigned    dst,
                   uint64_t    dstOffset,
                   void*       ctx)
{
  fi_rma_iov* iov = const_cast<fi_rma_iov*>(msg->rma_iov);
  iov->addr   += _ra[dst].addr;         // Revisit: Kinda kludgey
  iov->key     = _ra[dst].rkey;         // Revisit: _mr's?

  void* desc = _mr[dst]->desc();        // Revisit: Fix kluginess
  for (unsigned i = 0; i < msg->iov_count; ++i)
  {
    msg->desc[i] = desc;
  }

  msg->data    = iov->addr;
  msg->context = ctx;

  if (!_ep[dst]->write_msg(msg, FI_REMOTE_CQ_DATA)) // Revisit: Flags?
  {
    int errNum = _ep[dst]->error_num();
    fprintf(stderr, "%s() failed, ret = %d (%s)\n",
            "write_msg", errNum, _ep[dst]->error());
    return errNum;
  }

  //_ep[dst]->writeMsg(batch->iov(), batch->iovCount(), dstAddr, immData,
  //                   _ra[dst], _mr[dst]);

  return 0;
}
