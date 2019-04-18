#include "Server.hh"

#include "Header.hh"

#include <sys/socket.h>
#include <new>
#include <stdio.h>
#include <string.h>

using namespace Pds::Bld;

Server::Server(int fd) :
  _fd         (fd),
  _id         ( 0),
  _buffer     (new char[Header::MTU]),
  _buffer_size(         Header::MTU),
  _buffer_next(0)
{
}

void Server::setID  (uint32_t    id)
{
  _id = id;
}

void Server::publish(uint64_t    pulseId, 
                     uint64_t    timeStamp, 
                     const char* T,
                     unsigned    sizeofT)
{
  const Header& hdr = *reinterpret_cast<const Header*>(_buffer);
  if (_buffer_next) {
    new (_buffer+_buffer_next) Header(pulseId, timeStamp, hdr);
    _buffer_next += Header::sizeofNext;
  }
  else {
    new (_buffer) Header(pulseId, timeStamp, _id);
    _buffer_next = Header::sizeofFirst;
  }
  memcpy(_buffer+_buffer_next, T, sizeofT);
  _buffer_next += sizeofT;
  //
  //  Test if ready to send
  //
  if ((_buffer_next + sizeofT > _buffer_size) ||
      hdr.done(pulseId)) {
    ::send(_fd, _buffer, _buffer_next, 0);
    _buffer_next = 0;
  }
}

void Server::flush()
{
  if (_buffer_next) {
    ::send(_fd, _buffer, _buffer_next, 0);
    _buffer_next = 0;
  }
}

Server::~Server() {}

