#include "BldServer.hh"

#include "BldHeader.hh"

#include <sys/socket.h>
#include <new>
#include <stdio.h>

using namespace Bld;

template <class T>
BldServer<T>::BldServer(int fd, unsigned src) : 
  _fd         (fd),
  _src        (src),
  _buffer     (new char[8192]),
  _buffer_size(         8192),
  _buffer_next(0)
{
}

template <class T>
void BldServer<T>::publish(uint64_t pulseId, uint64_t timeStamp, const T& t)
{
  const BldHeader& hdr = *reinterpret_cast<const BldHeader*>(_buffer);
  if (_buffer_next) {
    new (_buffer+_buffer_next) BldHeader(pulseId, timeStamp, hdr);
    _buffer_next += BldHeader::sizeofNext;
  }
  else {
    new (_buffer) BldHeader(pulseId, timeStamp, _src);
    _buffer_next = BldHeader::sizeofFirst;
  }
  new (_buffer+_buffer_next) T(t);
  _buffer_next += sizeof(T);
  //
  //  Test if ready to send
  //
  if ((_buffer_next + sizeof(T) > _buffer_size) ||
      hdr.done(pulseId)) {
    ::send(_fd, _buffer, _buffer_next, 0);
    _buffer_next = 0;
  }
}

template <class T>
BldServer<T>::~BldServer() {}

